"""The vector CCIL transition model: an MLP residual dynamics model with Lipschitz regularization.

Ported from correct_il/models/nn_dynamics.py (WorldModel, DynamicsNet, the
Lipschitz losses, fit_model) at archive/upstream-main-2026-10-22; see
docs/migration/vector-dynamics.md for the function-by-function map and every
intentional difference. It implements ccil.core.transition.TransitionModel and
needs the ccil[torch] extra; the core never imports this module directly.

Kept from legacy:
- inputs are normalized by their mean and mean absolute deviation; the network
  predicts the residual s_{t+1} - s_t normalized the same way; output
  dimensions whose scale is below 1e-8 are masked to zero;
- losses: mean squared error on the normalized residual, plus the configured
  Lipschitz regularizer (spectral normalization, soft sampled or Jacobian
  penalties, and the slack variants with a learned per-sample weight);
- Adam, minibatches drawn by a fresh permutation each epoch, dropping the last
  partial batch.

Correctness changes from legacy (migration contract section 6; evidence in
docs/migration/vector-dynamics.md), which is why kind mlp is version 2:
- delta and evaluate run in eval mode and restore the prior mode, so
  prediction never updates spectral-norm power-iteration buffers and labels do
  not depend on call order (legacy never called eval()).
- every Lipschitz penalty constrains the residual model f, as the paper does
  (Ke et al. 2024, Sec. 4.2, Eqs. 3-4 and 10): soft sampling uses the
  finite-difference ratio ||f(x + dx) - f(x)|| / ||dx|| with a hinge at the
  constraint, and soft_jac the spectral norm of df/d(s, a). Legacy soft
  sampling compared the normalized residual with the next-state prediction and
  divided by the unscaled noise; legacy soft_jac constrained the next-state map
  s + f, whose Jacobian is I + df/ds.
- soft_sampling_slack weights each sample's perturbation penalties by that
  sample's own slack; legacy broadcast every slack against every sample's
  perturbations.
- the sampled penalties' coordinates are explicit (#125,
  transition_model.lipschitz_coordinates): raw, as above, or model_normalized,
  which perturbs the model's normalized inputs and compares its normalized
  residuals (a dimensionless ratio, invariant to units and timestep). The
  penalty is a training regularizer: lipschitz_constraint is never a certified
  bound, and certificates recompute bounds from the weights.
"""
from __future__ import annotations

import json
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..core.transition import check_identity, checkpoint_meta, one_step_metrics
from ..schema.io import _atomic_savez
from .spectral_norm import LipschitzSpectralNorm, apply_spectral_norm

DERIVED_FROM = {
    "repository": "personalrobotics/CCIL",
    "ref": "archive/upstream-main-2026-10-22",
    "commit": "477ddb81c7adc3527c77b094c08181a98077eff3",
    "files": ["correct_il/models/nn_dynamics.py", "correct_il/models/spectral_norm.py"],
    "paper": "Ke et al., CCIL: Continuity-Based Data Augmentation for Corrective Imitation Learning, ICLR 2024, "
             "arXiv:2310.12972",
}
LIPSCHITZ_TYPES = ("none", "spectral_normalization", "slack", "spectral_normalization_slack", "soft_sampling",
                   "soft_sampling_slack", "soft_jac")
SETTINGS = ("hidden_layers", "activation", "lipschitz_type", "lipschitz_constraint", "soft_penalty_weight",
            "soft_sampling_eps", "soft_samples", "slack_weight", "learning_rate", "weight_decay", "epochs",
            "batch_size", "lipschitz_coordinates")
LIPSCHITZ_COORDINATES = ("raw", "model_normalized")
LIPSCHITZ_UNITS = {
    "raw": "||f(s + ds, a + da) - f(s, a)|| / ||(ds, da)||: raw residuals over raw perturbations eps * N(0, I) (#21)",
    "model_normalized": "||g(x + dx) - g(x)|| / ||dx||: the model's normalized residual g over its normalized inputs "
                        "x = (q - shift) / (scale + 1e-8), perturbed by dx = eps * N(0, I) in that space (#125)",
}
PREDICT_CHUNK = 65536
JACOBIAN_MEMORY_BYTES = 2048 * 2**20  # default device-memory budget of one Jacobian chunk (compute.jacobian_memory_mb)


def jacobian_row_bytes(state_dim: int, action_dim: int, hidden_layers) -> int:
    """Estimated device bytes of one row of a vmap(jacrev) chunk.

    A row costs the activations reverse mode keeps for each of its D output directions, about 2 D sum(hidden)
    floats (measured on Pendulum's [256, 256] model, D = 3: 0.76 GB for 65,536 rows), and its float32 Jacobian,
    D (D + A) values, held about three times over at large D (the D x D cotangent basis, the result and its copy;
    measured at D = 512, A = 7). The estimate allows twice the activations and three Jacobians, plus one forward
    pass.
    """
    h = sum(int(x) for x in hidden_layers)
    return 4 * (3 * state_dim * (state_dim + action_dim) + 4 * state_dim * h + 2 * h + state_dim + action_dim)


class JacobianBudgetError(ValueError):
    """compute.jacobian_memory_mb is smaller than one row of the Jacobian computation."""


def jacobian_chunk_rows(state_dim: int, action_dim: int, hidden_layers, budget_bytes: int) -> int:
    """Rows per vmap(jacrev) call that fit a device-memory budget, falling as the state dimension and width grow.

    The budget is a bound, so a budget smaller than one estimated row fails closed (JacobianBudgetError, naming
    the minimum) rather than running a one-row chunk over it."""
    row = jacobian_row_bytes(state_dim, action_dim, hidden_layers)
    if row > int(budget_bytes):
        raise JacobianBudgetError(f"one Jacobian row needs about {row / 2**20:.2f} MiB (D = {state_dim}, A = "
                                  f"{action_dim}, hidden {list(hidden_layers)}), more than compute.jacobian_memory_mb "
                                  f"= {int(budget_bytes) / 2**20:.2f} MiB; raise it to at least {-(-row // 2**20)}")
    return int(budget_bytes) // row
NORMALIZATION = ("s_shift", "s_scale", "a_shift", "a_scale", "out_shift", "out_scale")


class DynamicsNet(nn.Module):
    """Normalized MLP; forward returns the normalized residual, predict the raw residual."""

    def __init__(self, state_dim, act_dim, hidden_layers, out_dim=None, activation="relu"):
        super().__init__()
        self.state_dim, self.act_dim = state_dim, act_dim
        self.out_dim = state_dim if out_dim is None else out_dim
        sizes = (state_dim + act_dim, *hidden_layers, self.out_dim)
        self.fc_layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])
        self.nonlinearity = torch.tanh if activation == "tanh" else torch.relu
        for name, dim in zip(NORMALIZATION, (state_dim, state_dim, act_dim, act_dim, self.out_dim, self.out_dim)):
            self.register_buffer(name, torch.zeros(dim) if name.endswith("shift") else torch.ones(dim))

    def enable_spectral_normalization(self, lipschitz_constraint):
        for layer in self.fc_layers:
            apply_spectral_norm(layer, lipschitz_constraint)

    def set_transformations(self, **tensors):
        for name in NORMALIZATION:
            getattr(self, name).copy_(tensors[name])

    def normalize(self, s, a):
        """The normalized input x = (q - shift) / (scale + 1e-8) of q = (s, a)."""
        return torch.cat([(s - self.s_shift) / (self.s_scale + 1e-8), (a - self.a_shift) / (self.a_scale + 1e-8)], -1)

    def forward_normalized(self, x):
        """The normalized residual g(x) of a normalized input."""
        out = x
        for layer in self.fc_layers[:-1]:
            out = self.nonlinearity(layer(out))
        return self.fc_layers[-1](out)

    def forward(self, s, a):
        return self.forward_normalized(self.normalize(s, a))

    def predict(self, s, a):
        out = self.forward(s, a) * (self.out_scale + 1e-8) + self.out_shift
        return out * (self.out_scale >= 1e-8)


# --- losses: (X, Y_pred, Y) -> dict with a differentiable "loss" ------------
# residual_fn is the model's raw residual f(s, a), in state units.

def mse_loss(_, y_pred, y):
    loss = F.mse_loss(y_pred, y)
    return {"loss": loss, "mean_error": (y_pred - y).norm(dim=-1).mean().detach()}


def lipschitz_ratio(s, a, residual_fn, n_samples, sampling_eps, noise=None, coordinates="raw", net=None):
    """Finite-difference Lipschitz estimates of the residual model, (B * n_samples,).

    Rows i * n_samples ... (i + 1) * n_samples - 1 belong to sample i. As in
    legacy, state and action are perturbed together. noise optionally supplies
    the unit normal draws (ns, na) with the repeated shapes. coordinates
    (LIPSCHITZ_UNITS):
    - raw: perturb (s, a) by sampling_eps * N(0, I) and give
      ||f(s + ds, a + da) - f(s, a)|| / ||(ds, da)||, in raw units;
    - model_normalized: perturb net's normalized input x by dx = sampling_eps *
      N(0, I) and give ||g(x + dx) - g(x)|| / ||dx||, with g the normalized
      residual: a dimensionless ratio, invariant to rescaling the raw units.
    """
    if coordinates == "model_normalized":
        x = net.normalize(s, a)
        base = net.forward_normalized(x).repeat_interleave(n_samples, dim=0)
        x = x.repeat_interleave(n_samples, dim=0)
        ns, na = noise if noise is not None else (torch.randn_like(s.repeat_interleave(n_samples, dim=0)),
                                                  torch.randn_like(a.repeat_interleave(n_samples, dim=0)))
        dx = torch.cat([ns, na], dim=-1) * sampling_eps
        return torch.linalg.norm(net.forward_normalized(x + dx) - base, dim=-1) / torch.linalg.norm(dx, dim=-1)
    base = residual_fn(s, a).repeat_interleave(n_samples, dim=0)
    s, a = s.repeat_interleave(n_samples, dim=0), a.repeat_interleave(n_samples, dim=0)
    ns, na = noise if noise is not None else (torch.randn_like(s), torch.randn_like(a))
    ds, da = ns * sampling_eps, na * sampling_eps
    step = torch.linalg.norm(torch.cat([ds, da], dim=-1), dim=-1)
    return torch.linalg.norm(residual_fn(s + ds, a + da) - base, dim=-1) / step


def soft_sampling_loss(x, y_pred, y, lipschitz_constraint, sampling_eps, penalty_weight, n_samples, residual_fn,
                       coordinates="raw", net=None):
    ratio = lipschitz_ratio(*x, residual_fn, n_samples, sampling_eps, coordinates=coordinates, net=net)
    penalty = penalty_weight * torch.relu(ratio - lipschitz_constraint).mean()
    mse = F.mse_loss(y_pred, y)
    return {"loss": mse + penalty, "mse_loss": mse.detach(), "lipschitz_penalty": penalty.detach()}


def slack_sampling_terms(x, residual_fn, lambda_fn, lipschitz_constraint, sampling_eps, n_samples, noise=None,
                         coordinates="raw", net=None):
    """Per-sample slack-weighted penalty, (B,): sample i's slack times the mean hinge over its own perturbations."""
    hinge = torch.relu(lipschitz_ratio(*x, residual_fn, n_samples, sampling_eps, noise, coordinates, net)
                       - lipschitz_constraint)
    slack = lambda_fn(*x).reshape(-1)
    return slack * hinge.reshape(len(slack), n_samples).mean(dim=1), slack


def soft_sampling_slack_loss(x, y_pred, y, lipschitz_constraint, sampling_eps, penalty_weight, n_samples,
                             residual_fn, lambda_fn, coordinates="raw", net=None):
    terms, slack = slack_sampling_terms(x, residual_fn, lambda_fn, lipschitz_constraint, sampling_eps, n_samples,
                                        coordinates=coordinates, net=net)
    # The last term is the legacy batch-level regularizer on slack magnitude, unchanged.
    penalty = penalty_weight * (terms.mean() + 1 - torch.exp(-0.1 * torch.abs(slack).mean()))
    mse = F.mse_loss(y_pred, y)
    return {"loss": mse + penalty, "mse_loss": mse.detach(), "lipschitz_penalty": penalty.detach()}


def slack_loss(x, y_pred, y, lambda_fn, slack_weight):
    slack = torch.sigmoid(lambda_fn(*x))
    mse = F.mse_loss(y_pred, y, reduction="none")
    weighted = torch.linalg.norm(slack * mse, dim=-1).mean()
    avg_slack = slack.mean()
    return {"loss": weighted - slack_weight * avg_slack, "mse_loss": mse.norm(dim=-1).mean().detach(),
            "weighted_mse_loss": weighted.detach(), "slack_penalty": avg_slack.detach(), "slack_std": slack.std().detach()}


def residual_jacobian_norm(s, a, residual_fn):
    """Spectral norm of df/d(s, a) per sample, (B,)."""
    d = s.shape[-1]
    jac = torch.func.vmap(torch.func.jacrev(lambda sa: residual_fn(sa[..., :d], sa[..., d:])))(torch.cat([s, a], -1))
    return torch.linalg.norm(jac, ord=2, dim=(-2, -1))


def soft_jac_loss(x, y_pred, y, lipschitz_constraint, penalty_weight, residual_fn):
    local_l = residual_jacobian_norm(*x, residual_fn)
    penalty = penalty_weight * torch.relu(local_l - lipschitz_constraint).mean()
    mse = F.mse_loss(y_pred, y)
    return {"loss": mse + penalty, "mse_loss": mse.detach(), "lipschitz_penalty": penalty.detach()}


def normalization_sha256(net) -> str:
    """SHA-256 of a model's normalization (shift and scale of state, action and residual), as float32 bytes."""
    import hashlib
    h = hashlib.sha256()
    for name in NORMALIZATION:
        h.update(name.encode() + getattr(net, name).detach().cpu().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def lipschitz_record(settings: dict, net) -> dict:
    """What a fit record states about its Lipschitz penalty's coordinates (#125); empty without a sampled one.

    The penalty is a training regularizer, never a certificate: lipschitz_constraint is not a proven bound in
    either coordinate system, and certificates recompute model bounds from the weights."""
    if "soft_sampling" not in settings["lipschitz_type"] and settings["lipschitz_type"] != "soft_jac":
        return {}
    c = settings.get("lipschitz_coordinates", "raw")
    return {"lipschitz": {"coordinates": c, "units": LIPSCHITZ_UNITS[c], "constraint": settings["lipschitz_constraint"],
                          "normalization_sha256": normalization_sha256(net),
                          "role": "training regularizer; not a certified bound"}}


class MLPTransitionModel:
    """TransitionModel kind "mlp": the legacy WorldModel on the core interface."""

    kind, version = "mlp", "2"
    evaluation_dtype = "float32"

    def __init__(self, hidden_layers, epochs, batch_size, activation="relu", lipschitz_type=None,
                 lipschitz_constraint=None, soft_penalty_weight=None, soft_sampling_eps=0.0003, soft_samples=5,
                 slack_weight=1.0, learning_rate=0.001, weight_decay=1e-05, lipschitz_coordinates="raw", device="cpu",
                 deterministic=False):
        lipschitz_type = lipschitz_type or "none"
        if lipschitz_type not in LIPSCHITZ_TYPES:
            raise ValueError(f"lipschitz_type {lipschitz_type!r} is not one of {LIPSCHITZ_TYPES}")
        if lipschitz_coordinates not in LIPSCHITZ_COORDINATES:
            raise ValueError(f"lipschitz_coordinates {lipschitz_coordinates!r} is not one of {LIPSCHITZ_COORDINATES}")
        if lipschitz_type == "soft_jac" and lipschitz_coordinates != "raw":
            raise ValueError("soft_jac has no model_normalized form yet (#125); use lipschitz_coordinates raw")
        self.settings = {"hidden_layers": [int(h) for h in hidden_layers], "activation": activation,
                         "lipschitz_type": lipschitz_type, "lipschitz_constraint": lipschitz_constraint,
                         "soft_penalty_weight": soft_penalty_weight, "soft_sampling_eps": soft_sampling_eps,
                         "soft_samples": soft_samples, "slack_weight": slack_weight, "learning_rate": learning_rate,
                         "weight_decay": weight_decay, "epochs": int(epochs), "batch_size": int(batch_size),
                         "lipschitz_coordinates": lipschitz_coordinates}
        self.device = _check_device(device)
        self.deterministic = bool(deterministic)
        self.jacobian_memory_bytes = JACOBIAN_MEMORY_BYTES
        self.jacobian_chunking = None  # the last jacobian_za call's chunking, for records
        self.net = self.lam = None

    @classmethod
    def from_config(cls, v):
        return cls(**{k: v[f"transition_model.{k}"] for k in SETTINGS},
                   device=v["compute.device"], deterministic=v["compute.deterministic"])

    # --- construction ------------------------------------------------------
    def _build(self, state_dim, act_dim):
        st = self.settings
        self.net = DynamicsNet(state_dim, act_dim, st["hidden_layers"], activation=st["activation"]).to(self.device)
        if "spectral_normalization" in st["lipschitz_type"]:
            self.net.enable_spectral_normalization(st["lipschitz_constraint"])
        self.lam = None
        if "slack" in st["lipschitz_type"]:
            self.lam = DynamicsNet(state_dim, act_dim, st["hidden_layers"], out_dim=1).to(self.device)

    def _loss_fn(self):
        st, lt = self.settings, self.settings["lipschitz_type"]
        lam = (lambda s, a: self.lam.predict(s, a)) if self.lam is not None else None
        coords = {"coordinates": st["lipschitz_coordinates"], "net": self.net}
        if lt == "soft_sampling":
            return partial(soft_sampling_loss, lipschitz_constraint=st["lipschitz_constraint"],
                           sampling_eps=st["soft_sampling_eps"], penalty_weight=st["soft_penalty_weight"],
                           n_samples=st["soft_samples"], residual_fn=self.net.predict, **coords)
        if lt == "soft_sampling_slack":
            return partial(soft_sampling_slack_loss, lipschitz_constraint=st["lipschitz_constraint"],
                           sampling_eps=st["soft_sampling_eps"], penalty_weight=st["soft_penalty_weight"],
                           n_samples=st["soft_samples"], residual_fn=self.net.predict, lambda_fn=lam, **coords)
        if lt in ("slack", "spectral_normalization_slack"):
            return partial(slack_loss, lambda_fn=lam, slack_weight=st["slack_weight"])
        if lt == "soft_jac":
            return partial(soft_jac_loss, lipschitz_constraint=st["lipschitz_constraint"],
                           penalty_weight=st["soft_penalty_weight"], residual_fn=self.net.predict)
        return mse_loss

    # --- TransitionModel ---------------------------------------------------
    def fit(self, z, a, z_next, seed: int = 0) -> dict:
        n, bs = len(z), self.settings["batch_size"]
        if n < bs:
            raise ValueError(f"{n} transitions is fewer than one minibatch (transition_model.batch_size={bs})")
        devices = [torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        with torch.random.fork_rng(devices=devices), _determinism(self.deterministic):
            torch.manual_seed(seed)
            self._build(z.shape[1], a.shape[1])
            s, act, sp = (torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device) for x in (z, a, z_next))
            s_shift, a_shift, out_shift = s.mean(0), act.mean(0), (sp - s).mean(0)
            self.net.set_transformations(s_shift=s_shift, s_scale=(s - s_shift).abs().mean(0), a_shift=a_shift,
                                         a_scale=(act - a_shift).abs().mean(0), out_shift=out_shift,
                                         out_scale=(sp - s - out_shift).abs().mean(0))
            y = (sp - s - self.net.out_shift) / (self.net.out_scale + 1e-8)
            params = list(self.net.parameters()) + (list(self.lam.parameters()) if self.lam is not None else [])
            opt = torch.optim.Adam(params, lr=self.settings["learning_rate"], weight_decay=self.settings["weight_decay"])
            loss_fn, order, steps = self._loss_fn(), np.random.default_rng(seed), n // bs
            history: dict = {}
            for _ in range(self.settings["epochs"]):
                perm = torch.as_tensor(order.permutation(n), device=self.device)
                totals: dict = {}
                for mb in range(steps):
                    idx = perm[mb * bs:(mb + 1) * bs]
                    x = (s[idx], act[idx])
                    opt.zero_grad()
                    out = loss_fn(x, self.net(*x), y[idx])
                    out["loss"].backward()
                    opt.step()
                    for k, val in out.items():
                        totals[k] = totals.get(k, 0.0) + float(val.detach())
                for k, total in totals.items():
                    history.setdefault(k, []).append(total / steps)
        return {"method": "adam", "epochs": self.settings["epochs"], "steps_per_epoch": steps,
                "final_epoch_losses": {k: v[-1] for k, v in history.items()}, "epoch_losses": history,
                **lipschitz_record(self.settings, self.net), **one_step_metrics(self, z, a, z_next)}

    def delta(self, z, a) -> np.ndarray:
        with torch.no_grad(), _determinism(self.deterministic), _eval_mode(self.net, self.lam):
            out = [self.net.predict(torch.as_tensor(np.asarray(z[i:i + PREDICT_CHUNK]), dtype=torch.float32, device=self.device),
                                    torch.as_tensor(np.asarray(a[i:i + PREDICT_CHUNK]), dtype=torch.float32, device=self.device))
                   for i in range(0, len(z), PREDICT_CHUNK)]
        if not out:
            return np.zeros((0, self.net.state_dim))
        return torch.cat(out).cpu().numpy().astype(np.float64)

    def jacobian_za(self, z, a) -> np.ndarray:
        """Exact df/d(z, a) in eval mode, (N, D, D + A).

        Rows are processed in balanced chunks of at most jacobian_chunk_rows(D, A, hidden, jacobian_memory_bytes),
        so peak device memory is bounded by the budget, not by N or D (#109); balanced sizes avoid a tiny final
        chunk, whose different kernel could move a pre-activation lying within rounding of a ReLU kink to the other
        side (the exact Jacobian of a ReLU network is discontinuous there). The chunking is kept in
        jacobian_chunking for records. The result is assembled on the host in float64.
        """
        d = self.net.state_dim
        z, a = np.asarray(z), np.asarray(a)
        out = np.empty((len(z), d, d + a.shape[1]), np.float64)
        jac_fn = torch.func.vmap(torch.func.jacrev(lambda sa: self.net.predict(sa[..., :d], sa[..., d:])))
        cap = jacobian_chunk_rows(d, a.shape[1], self.settings["hidden_layers"], self.jacobian_memory_bytes)
        chunks = max(1, -(-len(z) // cap))
        size = -(-len(z) // chunks) if len(z) else 1  # ceil(N / ceil(N / cap)): balanced chunks
        self.jacobian_chunking = {"budget_bytes": int(self.jacobian_memory_bytes), "max_rows_per_chunk": cap,
                                  "rows": int(len(z)), "chunks": chunks if len(z) else 0, "rows_per_chunk": size}
        with _determinism(self.deterministic), _eval_mode(self.net, self.lam):
            for i in range(0, len(z), size):
                x = torch.cat([torch.as_tensor(z[i:i + size], dtype=torch.float32, device=self.device),
                               torch.as_tensor(a[i:i + size], dtype=torch.float32, device=self.device)], -1)
                out[i:i + size] = jac_fn(x).detach().cpu().numpy()
        return out

    def jacobian_z(self, z, a) -> np.ndarray:
        """Exact df/dz in eval mode, (N, D, D); used by Newton solves instead of float32 finite differences."""
        return self.jacobian_za(z, a)[:, :, :self.net.state_dim]

    def evaluate(self, z, a, z_next) -> dict:
        return one_step_metrics(self, z, a, z_next)

    def lipschitz_bound(self, maps: str, state_scale, action_scale) -> dict:
        """A global Lipschitz bound of f_hat in state or action, in scaled coordinates (#88).

        f_hat = m * (out_scale + 1e-8) * g(x) + out_shift with x the normalized
        input and g = W_L act(... act(W_1 x + b_1) ...) + b_L, so with respect to
        the scaled input u (z = S_z u or a = S_a u) and the scaled error S_z^-1 e,

            K <= ||S_z^-1 M D_out W_L|| * ||W_{L-1}|| ... ||W_2|| * ||W_1[:, cols] D_in S|| * L_act^(L-1),

        with D_in = 1 / (scale + 1e-8) and D_out = out_scale + 1e-8 as the
        forward pass computes them in float32, M the output mask, and each norm the
        exact spectral norm (SVD in float64) of the weight the eval-mode forward
        applies: for spectral normalization, weight_orig / max(sigma_hat / c, 1)
        with the stored u, v, whose sigma_hat estimate and the constraint c are
        never used as a bound. ReLU and tanh are 1-Lipschitz; any other
        activation is refused. The value holds for the network in exact real
        arithmetic; neither the float32 evaluation of the network nor the
        float64 rounding of the norms is enclosed, so it is not "verified".
        """
        act = self.settings["activation"]
        if act not in ("relu", "tanh"):
            raise ValueError(f"no Lipschitz constant for activation {act!r}; supported: relu, tanh")
        net, d = self.net, self.net.state_dim
        sz, sa = np.asarray(state_scale, np.float64), np.asarray(action_scale, np.float64)
        with torch.no_grad(), _eval_mode(net):
            weights = [_effective_weight(layer).detach().cpu().double().numpy() for layer in net.fc_layers]
            f64 = lambda t: t.detach().cpu().double().numpy()  # noqa: E731
            in_div = f64(net.s_scale + 1e-8) if maps == "state" else f64(net.a_scale + 1e-8)
            out_mul = f64(net.out_scale + 1e-8) * f64(net.out_scale >= 1e-8)
        cols = slice(0, d) if maps == "state" else slice(d, d + net.act_dim)
        scale = sz if maps == "state" else sa
        first = weights[0][:, cols] * (scale / in_div)[None, :]
        rows = (out_mul / sz)[:, None]
        if len(weights) == 1:
            factors = [float(np.linalg.norm(rows * first, 2))]
        else:
            factors = ([float(np.linalg.norm(first, 2))] + [float(np.linalg.norm(w, 2)) for w in weights[1:-1]]
                       + [float(np.linalg.norm(rows * weights[-1], 2))])
        return {"value": float(np.prod(factors)), "factors": factors, "activation": act, "activation_lipschitz": 1.0,
                "verified": False}

    # --- persistence ---------------------------------------------------------
    def save(self, path) -> None:
        arrays = {f"dynamics/{k}": v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}
        if self.lam is not None:
            arrays.update({f"lambda/{k}": v.detach().cpu().numpy() for k, v in self.lam.state_dict().items()})
        meta = {"component": "transition_model", "kind": self.kind, "version": self.version,
                "state_dim": self.net.state_dim, "act_dim": self.net.act_dim, "settings": self.settings,
                "training": self.net.training, "dtype": "float32", "derived_from": DERIVED_FROM}
        _atomic_savez(path, {**arrays, "meta": np.array(json.dumps(meta, sort_keys=True))})

    @classmethod
    def load(cls, path, device: str = "cpu") -> "MLPTransitionModel":
        meta = checkpoint_meta(path, "transition_model")
        check_identity(path, meta, cls)
        model = cls(**meta["settings"], device=device)
        with torch.random.fork_rng(devices=[]):  # construction draws throwaway init values
            model._build(meta["state_dim"], meta["act_dim"])
        with np.load(path, allow_pickle=False) as z:
            for prefix, net in (("dynamics/", model.net), ("lambda/", model.lam)):
                if net is None:
                    continue
                state = {k[len(prefix):]: torch.from_numpy(z[k]) for k in z.files if k.startswith(prefix)}
                net.load_state_dict(state, strict=True)
            if model.lam is None and any(k.startswith("lambda/") for k in z.files):
                raise ValueError(f"{path} has a slack network but lipschitz_type {meta['settings']['lipschitz_type']!r}")
        for net in (model.net, model.lam):
            if net is not None:
                net.train(meta["training"])
        return model


def _effective_weight(layer) -> torch.Tensor:
    """The weight an eval-mode forward pass applies to layer."""
    for hook in layer._forward_pre_hooks.values():
        if isinstance(hook, LipschitzSpectralNorm):
            return hook.compute_weight(layer, do_power_iteration=False)
    return layer.weight


def _check_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"compute.device {device!r} requested but CUDA is not available")
    return str(device)


class _eval_mode:
    """Context manager: put the networks in eval mode, restoring each one's prior mode on exit."""

    def __init__(self, *nets):
        self.nets = [n for n in nets if n is not None]

    def __enter__(self):
        self.prev = [n.training for n in self.nets]
        for n in self.nets:
            n.eval()

    def __exit__(self, *exc):
        for n, mode in zip(self.nets, self.prev):
            n.train(mode)
        return False


class _determinism:
    """Context manager: torch.use_deterministic_algorithms(flag), restored on exit."""

    def __init__(self, flag: bool):
        self.flag = flag

    def __enter__(self):
        self.prev = torch.are_deterministic_algorithms_enabled()
        if self.flag:
            torch.use_deterministic_algorithms(True)

    def __exit__(self, *exc):
        torch.use_deterministic_algorithms(self.prev)
        return False
