"""The vector CCIL transition model: an MLP residual dynamics model with Lipschitz regularization.

Ported from correct_il/models/nn_dynamics.py (WorldModel, DynamicsNet, the
Lipschitz losses, fit_model) at archive/upstream-main-2026-10-22; see
docs/migration/vector-dynamics.md for the function-by-function map and every
intentional difference. It implements ccil.core.transition.TransitionModel and
needs the ccil[torch] extra; the core never imports this module directly.

The mathematics are unchanged:
- inputs are normalized by their mean and mean absolute deviation; the network
  predicts the residual s_{t+1} - s_t normalized the same way; output
  dimensions whose scale is below 1e-8 are masked to zero;
- losses: mean squared error on the normalized residual, plus the configured
  Lipschitz regularizer (spectral normalization, soft sampled or Jacobian
  penalties, and the slack variants with a learned per-sample weight);
- Adam, minibatches drawn by a fresh permutation each epoch, dropping the last
  partial batch.

Two correctness changes from legacy (migration contract section 6; evidence in
docs/migration/vector-dynamics.md), which is why kind mlp is version 2:
- delta and evaluate run in eval mode and restore the prior mode, so
  prediction never updates spectral-norm power-iteration buffers and labels do
  not depend on call order (legacy never called eval()).
- the soft-sampling penalty is the paper's finite-difference Lipschitz ratio of
  the residual model, ||f(x + dx) - f(x)|| / ||dx|| with a hinge at the
  constraint (Ke et al. 2024, Sec. 4.2, Eq. 10). Legacy compared the normalized
  residual with the next-state prediction and divided by the unscaled noise.
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
from .spectral_norm import apply_spectral_norm

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
            "batch_size")
PREDICT_CHUNK = 65536
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

    def forward(self, s, a):
        out = torch.cat([(s - self.s_shift) / (self.s_scale + 1e-8), (a - self.a_shift) / (self.a_scale + 1e-8)], -1)
        for layer in self.fc_layers[:-1]:
            out = self.nonlinearity(layer(out))
        return self.fc_layers[-1](out)

    def predict(self, s, a):
        out = self.forward(s, a) * (self.out_scale + 1e-8) + self.out_shift
        return out * (self.out_scale >= 1e-8)


# --- losses: (X, Y_pred, Y) -> dict with a differentiable "loss" ------------
# residual_fn is the model's raw residual f(s, a); predict_fn its next-state
# prediction s + f(s, a), which soft_jac constrains as in legacy.

def mse_loss(_, y_pred, y):
    loss = F.mse_loss(y_pred, y)
    return {"loss": loss, "mean_error": (y_pred - y).norm(dim=-1).mean().detach()}


def lipschitz_ratio(s, a, residual_fn, n_samples, sampling_eps):
    """Finite-difference Lipschitz estimates of the residual model, (B * n_samples,).

    Each sample perturbs (s, a) by sampling_eps * N(0, I) and returns
    ||f(s + ds, a + da) - f(s, a)|| / ||(ds, da)||, both in raw units. As in
    legacy, state and action are perturbed together.
    """
    base = residual_fn(s, a).repeat_interleave(n_samples, dim=0)
    s, a = s.repeat_interleave(n_samples, dim=0), a.repeat_interleave(n_samples, dim=0)
    ds, da = torch.randn_like(s) * sampling_eps, torch.randn_like(a) * sampling_eps
    step = torch.linalg.norm(torch.cat([ds, da], dim=-1), dim=-1)
    return torch.linalg.norm(residual_fn(s + ds, a + da) - base, dim=-1) / step


def soft_sampling_loss(x, y_pred, y, lipschitz_constraint, sampling_eps, penalty_weight, n_samples, residual_fn):
    ratio = lipschitz_ratio(*x, residual_fn, n_samples, sampling_eps)
    penalty = penalty_weight * torch.relu(ratio - lipschitz_constraint).mean()
    mse = F.mse_loss(y_pred, y)
    return {"loss": mse + penalty, "mse_loss": mse.detach(), "lipschitz_penalty": penalty.detach()}


def soft_sampling_slack_loss(x, y_pred, y, lipschitz_constraint, sampling_eps, penalty_weight, n_samples,
                             residual_fn, lambda_fn):
    penalty = torch.relu(lipschitz_ratio(*x, residual_fn, n_samples, sampling_eps) - lipschitz_constraint)
    slack = lambda_fn(*x)  # (B, 1) against (B * n,): broadcasts as in legacy; see the migration doc
    penalty = penalty_weight * (slack * penalty + 1 - torch.exp(-0.1 * torch.abs(slack).mean())).mean()
    mse = F.mse_loss(y_pred, y)
    return {"loss": mse + penalty, "mse_loss": mse.detach(), "lipschitz_penalty": penalty.detach()}


def slack_loss(x, y_pred, y, lambda_fn, slack_weight):
    slack = torch.sigmoid(lambda_fn(*x))
    mse = F.mse_loss(y_pred, y, reduction="none")
    weighted = torch.linalg.norm(slack * mse, dim=-1).mean()
    avg_slack = slack.mean()
    return {"loss": weighted - slack_weight * avg_slack, "mse_loss": mse.norm(dim=-1).mean().detach(),
            "weighted_mse_loss": weighted.detach(), "slack_penalty": avg_slack.detach(), "slack_std": slack.std().detach()}


def soft_jac_loss(x, y_pred, y, lipschitz_constraint, penalty_weight, predict_fn):
    s, a = x
    d = s.shape[-1]
    jac = torch.func.vmap(torch.func.jacrev(lambda sa: predict_fn(sa[..., :d], sa[..., d:])))(torch.cat([s, a], -1))
    local_l = torch.linalg.norm(jac, ord=2, dim=(-2, -1))
    penalty = penalty_weight * torch.relu(local_l - lipschitz_constraint).mean()
    mse = F.mse_loss(y_pred, y)
    return {"loss": mse + penalty, "mse_loss": mse.detach(), "lipschitz_penalty": penalty.detach()}


class MLPTransitionModel:
    """TransitionModel kind "mlp": the legacy WorldModel on the core interface."""

    kind, version = "mlp", "2"

    def __init__(self, hidden_layers, epochs, batch_size, activation="relu", lipschitz_type=None,
                 lipschitz_constraint=None, soft_penalty_weight=None, soft_sampling_eps=0.0003, soft_samples=5,
                 slack_weight=1.0, learning_rate=0.001, weight_decay=1e-05, device="cpu", deterministic=False):
        lipschitz_type = lipschitz_type or "none"
        if lipschitz_type not in LIPSCHITZ_TYPES:
            raise ValueError(f"lipschitz_type {lipschitz_type!r} is not one of {LIPSCHITZ_TYPES}")
        self.settings = {"hidden_layers": [int(h) for h in hidden_layers], "activation": activation,
                         "lipschitz_type": lipschitz_type, "lipschitz_constraint": lipschitz_constraint,
                         "soft_penalty_weight": soft_penalty_weight, "soft_sampling_eps": soft_sampling_eps,
                         "soft_samples": soft_samples, "slack_weight": slack_weight, "learning_rate": learning_rate,
                         "weight_decay": weight_decay, "epochs": int(epochs), "batch_size": int(batch_size)}
        self.device = _check_device(device)
        self.deterministic = bool(deterministic)
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

    def _predict_next(self, s, a):
        return s + self.net.predict(s, a)

    def _loss_fn(self):
        st, lt = self.settings, self.settings["lipschitz_type"]
        lam = (lambda s, a: self.lam.predict(s, a)) if self.lam is not None else None
        if lt == "soft_sampling":
            return partial(soft_sampling_loss, lipschitz_constraint=st["lipschitz_constraint"],
                           sampling_eps=st["soft_sampling_eps"], penalty_weight=st["soft_penalty_weight"],
                           n_samples=st["soft_samples"], residual_fn=self.net.predict)
        if lt == "soft_sampling_slack":
            return partial(soft_sampling_slack_loss, lipschitz_constraint=st["lipschitz_constraint"],
                           sampling_eps=st["soft_sampling_eps"], penalty_weight=st["soft_penalty_weight"],
                           n_samples=st["soft_samples"], residual_fn=self.net.predict, lambda_fn=lam)
        if lt in ("slack", "spectral_normalization_slack"):
            return partial(slack_loss, lambda_fn=lam, slack_weight=st["slack_weight"])
        if lt == "soft_jac":
            return partial(soft_jac_loss, lipschitz_constraint=st["lipschitz_constraint"],
                           penalty_weight=st["soft_penalty_weight"], predict_fn=self._predict_next)
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
                **one_step_metrics(self, z, a, z_next)}

    def delta(self, z, a) -> np.ndarray:
        with torch.no_grad(), _determinism(self.deterministic), _eval_mode(self.net, self.lam):
            out = [self.net.predict(torch.as_tensor(np.asarray(z[i:i + PREDICT_CHUNK]), dtype=torch.float32, device=self.device),
                                    torch.as_tensor(np.asarray(a[i:i + PREDICT_CHUNK]), dtype=torch.float32, device=self.device))
                   for i in range(0, len(z), PREDICT_CHUNK)]
        if not out:
            return np.zeros((0, self.net.state_dim))
        return torch.cat(out).cpu().numpy().astype(np.float64)

    def evaluate(self, z, a, z_next) -> dict:
        return one_step_metrics(self, z, a, z_next)

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
