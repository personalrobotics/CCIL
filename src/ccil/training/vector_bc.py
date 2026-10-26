"""Experimental vector-policy trainer: the reference consumer of CCIL supervision (#89).

Reference infrastructure, not the CCIL product: it reproduces the paper's BC,
NoiseBC, and CCIL vector-policy arms (#24) and consumes the trainer-neutral
supervision contract (ccil.supervision) like any other trainer. It needs the
ccil[torch] extra and is never imported by ccil labels generate.

The protocol is the legacy one, read from archive/upstream-main-2026-10-22
(train_bc_policy.py, CustomBC.py, utils.py) and the pinned d3rlpy fork
(personalrobotics/d3rlpy@e89a7da): a deterministic BC regressor, d3rlpy's
default vector encoder (hidden layers with ReLU after each), a linear output
and tanh; a standard observation scaler (x - mean) / (std + 1e-3) with the
population std, and a min_max action scaler to [-1, 1] with 1e-8, both fitted
on every training row; Adam (betas 0.9, 0.999, eps 1e-8, no weight decay);
every epoch reshuffles the rows and takes floor(N / batch_size) minibatches.

Declared differences from legacy:
- loss: sum_i w_i mean_j (pi(s_i) - a_i)_j^2 / sum_i w_i over the minibatch, in
  scaled action space; with unit weights this is legacy F.mse_loss. A
  minibatch whose weights sum to 0 takes no optimizer step (the step still
  counts).
- NoiseBC perturbs the standardized observations of each minibatch, as legacy
  did after its scaler, but through ccil.training.noise_bc: a counter-keyed
  draw per (seed, optimizer step, sample identity, draw index), returned as a
  new array; legacy wrote noise into the caller's batch in place.
- randomness is explicit: initialization under torch.manual_seed(seed) in a
  forked RNG, epoch e's permutation from SeedSequence([seed, e]), and NoiseBC
  from its counter hash, so resuming needs no RNG state.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
import torch.nn as nn

from ..core.transition import CheckpointError, checkpoint_meta
from ..schema.io import _atomic_savez
from .noise_bc import NoiseTrace

NAME, VERSION = "vector_mlp", "1"
OBS_EPS, ACT_EPS = 1e-3, 1e-8          # d3rlpy StandardScaler eps; MinMaxActionScaler epsilon
ADAM = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
SCALERS = {"observation": ("standard", "none"), "action": ("min_max", "none")}


class TrainingError(ValueError):
    """The supervision or settings cannot be trained as declared."""


class VectorBCPolicy:
    """A deterministic vector policy: scalers, MLP with ReLU, linear output, tanh."""

    name, version = NAME, VERSION

    def __init__(self, obs_dim: int, act_dim: int, hidden_layers, observation_scaler="standard", action_scaler="min_max"):
        if observation_scaler not in SCALERS["observation"] or action_scaler not in SCALERS["action"]:
            raise TrainingError(f"scalers must be observation {SCALERS['observation']} and action {SCALERS['action']}")
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.hidden_layers = [int(h) for h in hidden_layers]
        self.observation_scaler, self.action_scaler = observation_scaler, action_scaler
        sizes = (self.obs_dim, *self.hidden_layers)
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-1], self.act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.obs_mean = self.obs_std = self.act_min = self.act_max = None

    # --- scalers (d3rlpy formulas) ------------------------------------------
    def fit_scalers(self, obs: np.ndarray, act: np.ndarray) -> None:
        """Fit on every training row, unweighted, in float64 as d3rlpy accumulates them."""
        if self.observation_scaler == "standard":
            o = np.asarray(obs, np.float64)
            self.obs_mean = o.mean(axis=0)
            self.obs_std = np.sqrt(((o - self.obs_mean) ** 2).mean(axis=0))
        if self.action_scaler == "min_max":
            self.act_min, self.act_max = np.asarray(act).min(axis=0), np.asarray(act).max(axis=0)
        elif np.abs(act).max() > 1:
            raise TrainingError("action_scaler none needs actions in [-1, 1]: the policy output is tanh")

    def scale_obs(self, x: torch.Tensor) -> torch.Tensor:
        if self.observation_scaler == "none":
            return x
        mean = torch.tensor(self.obs_mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self.obs_std, dtype=torch.float32, device=x.device)
        return (x - mean) / (std + OBS_EPS)

    def scale_act(self, a: torch.Tensor) -> torch.Tensor:
        if self.action_scaler == "none":
            return a
        lo = torch.tensor(self.act_min, dtype=torch.float32, device=a.device)
        hi = torch.tensor(self.act_max, dtype=torch.float32, device=a.device)
        return ((a - lo) / (hi - lo + ACT_EPS)) * 2.0 - 1.0

    def unscale_act(self, a: torch.Tensor) -> torch.Tensor:
        if self.action_scaler == "none":
            return a
        lo = torch.tensor(self.act_min, dtype=torch.float32, device=a.device)
        hi = torch.tensor(self.act_max, dtype=torch.float32, device=a.device)
        return ((hi - lo + ACT_EPS) * ((a + 1.0) / 2.0)) + lo

    # --- inference: the evaluation interface (#24) ----------------------------
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Raw actions (N, A) for raw observations (N, D); eval mode, no gradient, inputs untouched."""
        device = next(self.net.parameters()).device
        with torch.no_grad():
            x = torch.as_tensor(np.array(obs, np.float32), device=device)
            return self.unscale_act(self.net(self.scale_obs(x))).cpu().numpy().astype(np.float64)

    def to(self, device: str) -> "VectorBCPolicy":
        self.net.to(device)
        return self

    # --- persistence: no executable serialization ----------------------------
    def save(self, path, optimizer=None, epoch: int = 0, meta: dict | None = None) -> None:
        arrays = {f"net/{k}": v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}
        for name in ("obs_mean", "obs_std", "act_min", "act_max"):
            if getattr(self, name) is not None:
                arrays[f"scaler/{name}"] = np.asarray(getattr(self, name))
        if optimizer is not None:
            for i, p in enumerate(self.net.parameters()):
                state = optimizer.state.get(p, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        arrays[f"adam/{i}/{key}"] = state[key].detach().cpu().numpy()
                if "step" in state:
                    arrays[f"adam/{i}/step"] = np.asarray(float(state["step"]), np.float64)
        record = {"component": "policy", "trainer": self.name, "version": self.version, "obs_dim": self.obs_dim,
                  "act_dim": self.act_dim, "hidden_layers": self.hidden_layers,
                  "observation_scaler": self.observation_scaler, "action_scaler": self.action_scaler,
                  "epoch": int(epoch), **(meta or {})}
        arrays["meta"] = np.array(json.dumps(record, sort_keys=True))
        _atomic_savez(path, arrays)

    @classmethod
    def load(cls, path, device: str = "cpu") -> tuple["VectorBCPolicy", dict, dict]:
        """(policy, meta, optimizer arrays) from a checkpoint; CheckpointError if it is another trainer's."""
        meta = checkpoint_meta(path, "policy")
        if (meta.get("trainer"), meta.get("version")) != (NAME, VERSION):
            raise CheckpointError(f"{path} holds a {meta.get('trainer')!r} version {meta.get('version')!r} policy, "
                                  f"not {NAME!r} version {VERSION!r}")
        policy = cls(meta["obs_dim"], meta["act_dim"], meta["hidden_layers"], meta["observation_scaler"],
                     meta["action_scaler"])
        with np.load(path, allow_pickle=False) as z:
            policy.net.load_state_dict({k[4:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("net/")})
            for name in ("obs_mean", "obs_std", "act_min", "act_max"):
                if f"scaler/{name}" in z.files:
                    setattr(policy, name, z[f"scaler/{name}"])
            adam = {k: z[k] for k in z.files if k.startswith("adam/")}
        return policy.to(device), meta, adam


def _optimizer(policy, learning_rate, adam_arrays=None):
    opt = torch.optim.Adam(policy.net.parameters(), lr=learning_rate, **ADAM)
    if adam_arrays:
        device = next(policy.net.parameters()).device
        for i, p in enumerate(policy.net.parameters()):
            if f"adam/{i}/exp_avg" in adam_arrays:
                opt.state[p] = {"step": torch.tensor(float(adam_arrays[f"adam/{i}/step"])),
                                "exp_avg": torch.as_tensor(adam_arrays[f"adam/{i}/exp_avg"], device=device).clone(),
                                "exp_avg_sq": torch.as_tensor(adam_arrays[f"adam/{i}/exp_avg_sq"], device=device).clone()}
    return opt


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """sum_i w_i mean_j (pred - target)_ij^2 / sum_i w_i; unit weights give F.mse_loss."""
    per_row = ((pred - target) ** 2).mean(dim=1)
    return (w * per_row).sum() / w.sum()


def training_rows(supervision, balance_labels: int | None):
    """Row indices into the supervision, with legacy balance_data repetition of the demonstrations."""
    demo = np.flatnonzero(supervision.source == "demonstration")
    syn = np.flatnonzero(supervision.source == "synthetic")
    if not balance_labels:
        return np.arange(len(supervision.weights))
    # Legacy: the labels, then the expert data once and num_labels more times.
    return np.concatenate([syn] + [demo] * (int(balance_labels) + 1))


def train_vector_bc(supervision, settings: dict, *, seed: int, device: str = "cpu", deterministic: bool = False,
                    noise=None, resume=None, on_epoch=None):
    """Train on supervision.arrays(); return (policy, record).

    settings: hidden_layers, learning_rate, batch_size, epochs,
    observation_scaler, action_scaler, balance_labels (legacy num_labels, or
    None). noise: a NoiseBC, for the noise_bc arm only. resume: a checkpoint
    path written by on_epoch's policy.save at an epoch boundary. on_epoch:
    called as on_epoch(policy, optimizer, epoch, record) after each epoch.
    """
    obs, act, w = supervision.arrays()   # refuses chunked supervision (the contract)
    method = supervision.manifest()["method"]
    if noise is not None and method != "noise_bc":
        raise TrainingError(f"NoiseBC perturbation applies to the noise_bc arm only, not {method}")
    if method == "noise_bc" and noise is None:
        raise TrainingError("the noise_bc arm needs a NoiseBC perturbation")
    balance = settings.get("balance_labels")
    if balance and method != "ccil":
        raise TrainingError("balance_labels repeats demonstrations against labels; it applies to ccil only")
    rows = training_rows(supervision, balance)
    epochs = int(settings["epochs"]) // int(balance) if balance else int(settings["epochs"])
    if epochs < 1:
        raise TrainingError(f"{settings['epochs']} epochs divided by balance_labels {balance} leaves no epoch")
    bs = int(settings["batch_size"])
    n = len(rows)
    steps_per_epoch = n // bs
    if steps_per_epoch < 1:
        raise TrainingError(f"{n} training rows is fewer than one minibatch (batch_size {bs})")
    obs_all, act_all, w_all = np.asarray(obs, np.float32), np.asarray(act, np.float32), np.asarray(w, np.float32)

    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        prev = torch.are_deterministic_algorithms_enabled()
        if deterministic:
            torch.use_deterministic_algorithms(True)
        try:
            if resume is not None:
                policy, meta, adam = VectorBCPolicy.load(resume, device)
                start, digests = int(meta["epoch"]), list(meta.get("noise_trace_epochs", []))
                epoch_records = list(meta.get("epoch_records", []))
            else:
                torch.manual_seed(int(seed))
                policy = VectorBCPolicy(obs_all.shape[1], act_all.shape[1], settings["hidden_layers"],
                                        settings["observation_scaler"], settings["action_scaler"])
                policy.fit_scalers(obs_all[rows], act_all[rows])
                policy.to(device)
                adam, start, digests, epoch_records = None, 0, [], []
            opt = _optimizer(policy, float(settings["learning_rate"]), adam)
            obs_t = torch.as_tensor(obs_all, device=device)
            act_t = torch.as_tensor(act_all, device=device)
            w_t = torch.as_tensor(w_all, device=device)
            source = np.asarray(supervision.source)
            for epoch in range(start, epochs):
                policy.net.train()
                perm = rows[np.random.default_rng(np.random.SeedSequence([int(seed), epoch])).permutation(n)]
                trace = NoiseTrace() if noise is not None else None
                losses, seen = [], {"demonstration": 0, "synthetic": 0}
                for b in range(steps_per_epoch):
                    step = epoch * steps_per_epoch + b
                    idx = perm[b * bs:(b + 1) * bs]
                    x = policy.scale_obs(obs_t[idx])
                    if noise is not None:
                        batch = x.detach().cpu().numpy()
                        perturbed, rec = noise.perturb(batch, supervision.episode_id[idx], supervision.frame[idx], step, 0)
                        trace.update(step, 0, supervision.episode_id[idx], supervision.frame[idx], rec)
                        x = torch.as_tensor(perturbed, device=device)
                    wb = w_t[idx]
                    if float(wb.sum()) > 0:
                        loss = weighted_loss(policy.net(x), policy.scale_act(act_t[idx]), wb)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        losses.append(float(loss.detach()))
                    for s, c in zip(*np.unique(source[idx], return_counts=True)):
                        seen[str(s)] += int(c)
                if trace is not None:
                    digests.append(trace.summary()["sha256"])
                epoch_records.append({"epoch": epoch, "steps": steps_per_epoch, "mean_loss": float(np.mean(losses)) if losses else None,
                                      "rows_seen": seen, "zero_weight_steps": steps_per_epoch - len(losses)})
                if on_epoch is not None:
                    on_epoch(policy, opt, epoch + 1, {"noise_trace_epochs": digests, "epoch_records": epoch_records})
        finally:
            torch.use_deterministic_algorithms(prev)
    policy.net.eval()
    record = {"trainer": NAME, "version": VERSION, "settings": {**settings, "hidden_layers": list(settings["hidden_layers"])},
              "seed": int(seed), "device": str(device), "deterministic": bool(deterministic),
              "rows": int(n), "steps_per_epoch": steps_per_epoch, "rows_dropped_per_epoch": int(n % bs),
              "epochs_run": epochs, "balance_labels": balance or None,
              "row_sources": {"demonstration": int((source[rows] == "demonstration").sum()),
                              "synthetic": int((source[rows] == "synthetic").sum())},
              "epochs": epoch_records,
              "loss": "sum_i w_i mean_j (pi(s_i) - a_i)_j^2 / sum_i w_i per minibatch, scaled action space",
              "optimizer": {"name": "adam", "learning_rate": float(settings["learning_rate"]),
                            "betas": list(ADAM["betas"]), "eps": ADAM["eps"], "weight_decay": ADAM["weight_decay"]},
              "noise_bc": None if noise is None else {
                  **noise.descriptor(), "trace_epochs": digests,
                  "trace_sha256": hashlib.sha256("".join(digests).encode()).hexdigest(),
                  "applied_to": "standardized observations of each minibatch (after the observation scaler)"}}
    return policy, record


__all__ = ["VectorBCPolicy", "train_vector_bc", "weighted_loss", "training_rows", "TrainingError", "NAME", "VERSION"]
