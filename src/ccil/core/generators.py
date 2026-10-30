"""Generators propose corrective-label candidates from transitions and a model.

Contract
- generate(batch, model, seed) returns a CandidateBatch for every transition; it
  never accepts or rejects (validators do) but marks failed candidates.
- Generators are stateless: everything they use is in params, and params fully
  determine the output given (batch, model, seed).
- Randomness is drawn per transition from default_rng seeded with run.seed and
  a 128-bit digest of (episode_id, frame) (ccil.core.identity), so splitting a
  batch into chunks never changes any candidate and distinct transitions never
  share a stream.
- Measurements describe the candidate as stored: they are computed from the
  final, dtype-cast candidate arrays, and a solved candidate's convergence is
  judged on its stored values.
- name/version/params identify the generator in the candidate provenance, and
  params are written into the candidate file.
- label_error_profile names the CCIL label-error definition its labels follow
  (backtrack or disturbed_action); a certificate profile names the families it
  certifies (src/ccil/core/certificates.py).
- measurement_specs declares, for each measurement the generator returns, its
  definition, norm, and units; generate_labels writes them into the candidate
  file under the "generator." prefix it gives the measurements.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .batch import CandidateBatch, TransitionBatch
from .identity import identity_words32
from .transition import jacobian_fd


@runtime_checkable
class Generator(Protocol):
    name: str
    version: str
    params: dict

    def generate(self, batch: TransitionBatch, model, seed: int) -> CandidateBatch: ...


def transition_rng(seed: int, episode_id: str, frame: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), *identity_words32(episode_id, frame)])


REP = "representation units (state units for representation.kind=identity)"
SPEC_DISPLACEMENT = {"definition": "||z_G - z_t||: distance of the label state from the demonstration state of its "
                                   "source transition", "norm": "l2", "units": REP}
SPEC_SOLVER = {"definition": "||z_G + f(z_G, a_G) - target|| of the returned Newton iterate under the model",
               "norm": "l2", "units": REP}


class BackwardEuler:
    """z_G = z - f(z, a): applying a at z_G approximately reaches z."""

    name, version = "backward_euler", "1"
    label_error_profile = "backtrack"  # Ke et al., Def. 4.5
    measurement_specs: dict = {}

    def __init__(self):
        self.params = {}

    def generate(self, batch, model, seed):
        n = len(batch)
        return CandidateBatch(source=np.arange(n), z=batch.z - model.delta(batch.z, batch.a), a=batch.a.copy(),
                              target=batch.z.copy(), failed=np.zeros(n, bool))


class ExactPreimage:
    """Solve z_G + f(z_G, a) = z by Newton's method, row by row.

    A row stops iterating once its own residual is within tolerance, so its
    result never depends on the other rows in the batch. A row whose Newton
    system is singular, whose iterate becomes non-finite, or which does not
    converge is marked failed; the other rows are unaffected.
    """

    name, version = "exact_preimage", "1"
    label_error_profile = "backtrack"  # Ke et al., Def. 4.5
    measurement_specs = {"solver_residual": SPEC_SOLVER}

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-4):
        self.params = {"max_iterations": int(max_iterations), "tolerance": float(tolerance)}

    def generate(self, batch, model, seed):
        z, failed, resid = newton_preimage(model, batch.z, batch.a, batch.z, out_dtype=batch.z.dtype, **self.params)
        return CandidateBatch(source=np.arange(len(batch)), z=z, a=batch.a.copy(), target=batch.z.copy(), failed=failed,
                              measurements={"solver_residual": resid})


def newton_preimage(model, target, a, init, max_iterations, tolerance, out_dtype):
    """Solve x + f(x, a) = target row by row from init; return (x as out_dtype, failed, residual).

    See ExactPreimage for the per-row stopping and failure rules. The returned
    residual and failure status are those of the stored candidate: x is cast to
    out_dtype first and its residual recomputed, so a cast that moves an
    iterate outside the tolerance marks that row failed.
    """
    target, a = np.asarray(target, np.float64), np.asarray(a, np.float64)
    x = np.array(init, np.float64)
    eye = np.eye(x.shape[1])
    broken = np.zeros(len(x), bool)
    resid = np.linalg.norm(x + model.delta(x, a) - target, axis=1) if len(x) else np.zeros(0)
    for _ in range(max_iterations):
        idx = np.flatnonzero(~broken & ~(resid <= tolerance))
        if not len(idx):
            break
        xs, as_ = x[idx], a[idx]
        r = xs + model.delta(xs, as_) - target[idx]
        J = eye + (model.jacobian_z(xs, as_) if hasattr(model, "jacobian_z") else jacobian_fd(model, xs, as_))
        step, ok = _solve_rows(J, r)
        ok &= np.isfinite(xs - step).all(axis=1)
        x[idx[ok]] = xs[ok] - step[ok]
        broken[idx[~ok]] = True
        moved = idx[ok]
        resid[moved] = np.linalg.norm(x[moved] + model.delta(x[moved], a[moved]) - target[moved], axis=1)
    # Convergence is judged on the stored candidate, not on the float64 iterate or a stale residual.
    stored = x.astype(out_dtype)
    resid = np.linalg.norm(stored + model.delta(stored, a) - target, axis=1) if len(x) else np.zeros(0)
    return stored, broken | ~(resid <= tolerance), resid


def _solve_rows(J, r):
    """Solve J[i] s[i] = r[i]; rows with a singular or non-finite system get ok=False.

    Stacked LAPACK calls (the same gesv per system as a per-row call). A singular system makes numpy refuse its
    whole stack, so a refused stack is bisected until the singular rows are isolated: k singular rows cost
    O(k log N) stacked calls, and every other row gets the same step as a per-row solve.
    """
    ok = np.isfinite(J).all(axis=(1, 2)) & np.isfinite(r).all(axis=1)
    step = np.zeros_like(r)

    def solve(idx):
        try:
            step[idx] = np.linalg.solve(J[idx], r[idx][..., None])[..., 0]
        except np.linalg.LinAlgError:
            if len(idx) == 1:
                ok[idx[0]] = False
            else:
                solve(idx[:len(idx) // 2])
                solve(idx[len(idx) // 2:])
    idx = np.flatnonzero(ok)
    if len(idx):
        solve(idx)
    return step, ok & np.isfinite(step).all(axis=1)


SPEC_NOISE = {"definition": "||a_G - a_t||: norm of the sampled action perturbation", "norm": "l2",
              "units": "action units"}
SPEC_DRAW = {"definition": "ordinal j of the draw within its transition's stream (0 <= j < k); the first k of a "
                           "k_max generation are a generation at k", "norm": None, "units": "draw index"}
SPEC_ATTEMPT = {"definition": "raw index of the proposal in its transition's stream; with action bounds, proposals "
                              "whose stored action leaves the bounds are skipped, so the k draws are the first k "
                              "feasible proposals", "norm": None, "units": "attempt index"}
PROPOSAL_BLOCK = 64  # bounded proposals are drawn in fixed blocks, so a stream's prefix is the same at every k
MAX_ATTEMPTS = 1 << 20


def _bounds(action_bounds):
    if action_bounds is None:
        return None
    low, high = (np.asarray(b, np.float64).reshape(-1) for b in action_bounds)
    if low.shape != high.shape or not (np.isfinite(low).all() and np.isfinite(high).all() and (low < high).all()):
        raise ValueError("action_bounds must be finite (low, high) with low < high in every dimension")
    return low, high


def _noisy_actions(batch, seed, k, sigma, bounds=None):
    """k draws a~ = a + N(0, sigma^2 I) per transition, from transition_rng.

    With bounds (low, high), the k draws are the first k proposals of the stream whose stored a~ lies inside the
    bounds: a Gaussian truncated to the actuator range, so at a saturated action the perturbations point inward.
    Returns (source, draw ordinal, attempt index or None, a~ in the action dtype, ||a~ - a||) with the norm taken
    on the stored a~, so it is the perturbation the candidate actually carries (which rounding can shrink, even to
    zero, at large action magnitudes).
    """
    A = batch.a.shape[1]
    source = np.repeat(np.arange(len(batch)), k)
    draw = np.tile(np.arange(k, dtype=np.float64), len(batch))
    if bounds is None:
        noise = np.concatenate([transition_rng(seed, e, f).normal(0.0, sigma, size=(k, A))
                                for e, f in zip(batch.episode_id, batch.frame)]) if len(batch) else np.zeros((0, A))
        a_tilde = (batch.a[source] + noise).astype(batch.a.dtype)
        attempt = None
    else:
        low, high = bounds
        a64 = batch.a.astype(np.float64)
        if ((a64 < low) | (a64 > high)).any():
            raise ValueError("a demonstration action lies outside action_bounds, so no proposal around it is feasible")
        rows, attempts = [], []
        for i, (e, f) in enumerate(zip(batch.episode_id, batch.frame)):
            rng, got, idx, n = transition_rng(seed, e, f), [], [], 0
            while sum(map(len, got)) < k:
                if n >= MAX_ATTEMPTS:
                    raise ValueError(f"fewer than {k} feasible proposals in {MAX_ATTEMPTS} attempts at transition "
                                     f"({e}, {f}); sigma is far too large for the action bounds")
                cand = (a64[i] + rng.normal(0.0, sigma, size=(PROPOSAL_BLOCK, A))).astype(batch.a.dtype)
                ok = ((cand >= low) & (cand <= high)).all(axis=1)
                got.append(cand[ok])
                idx.append(n + np.flatnonzero(ok))
                n += PROPOSAL_BLOCK
            rows.append(np.concatenate(got)[:k])
            attempts.append(np.concatenate(idx)[:k])
        a_tilde = np.concatenate(rows) if rows else np.zeros((0, A), batch.a.dtype)
        attempt = np.concatenate(attempts).astype(np.float64) if attempts else np.zeros(0)
    stored_noise = a_tilde.astype(np.float64) - batch.a[source].astype(np.float64)
    return source, draw, attempt, a_tilde, np.linalg.norm(stored_noise, axis=1)


def _noise_params(k, sigma, action_bounds):
    params = {"labels_per_transition": int(k), "action_noise_std": float(sigma),
              "action_noise": "normal(0, action_noise_std^2) per action dimension, action units"}
    if action_bounds is not None:
        low, high = _bounds(action_bounds)
        params |= {"action_bounds": [low.tolist(), high.tolist()],
                   "action_noise": "normal(0, action_noise_std^2) per action dimension, action units, truncated to "
                                   "action_bounds by skipping proposals outside them"}
    return params


def _noise_measurements(attempt, **m):
    return m if attempt is None else {**m, "attempt_index": attempt}


def _displacement(z, origin):
    return np.linalg.norm(z.astype(np.float64) - origin.astype(np.float64), axis=1)


class NoisyAction:
    """For k draws a~ = a + N(0, sigma^2): z_G = z_next - f(z_next, a~), targeting z_next."""

    name, version = "noisy_action", "3"
    label_error_profile = "disturbed_action"  # Ke et al., Def. 4.6
    measurement_specs = {"action_noise_norm": SPEC_NOISE, "state_displacement_norm": SPEC_DISPLACEMENT,
                         "draw_index": SPEC_DRAW}

    def __init__(self, labels_per_transition: int, action_noise_std: float, action_bounds=None):
        self.params = _noise_params(labels_per_transition, action_noise_std, action_bounds)
        if action_bounds is not None:
            self.measurement_specs = {**type(self).measurement_specs, "attempt_index": SPEC_ATTEMPT}

    def generate(self, batch, model, seed):
        p = self.params
        source, draw, attempt, a_tilde, noise_norm = _noisy_actions(
            batch, seed, p["labels_per_transition"], p["action_noise_std"], _bounds(p.get("action_bounds")))
        target = batch.z_next[source]
        z = target - model.delta(target, a_tilde)
        return CandidateBatch(source=source, z=z, a=a_tilde, target=target.copy(), failed=np.zeros(len(source), bool),
                              measurements=_noise_measurements(attempt, action_noise_norm=noise_norm, draw_index=draw,
                                                               state_displacement_norm=_displacement(z, batch.z[source])))


class NoisyActionPreimage:
    """For k draws a~ = a + N(0, sigma^2): solve z_G + f(z_G, a~) = z_next by Newton's method from z.

    The legacy noisy_action_fe. A failed proposal (singular system, non-finite
    iterate, or no convergence) is marked failed and rejected; the k proposals
    per transition are never refilled to reach a count.
    """

    name, version = "noisy_action_preimage", "2"
    label_error_profile = "disturbed_action"  # Ke et al., Def. 4.6
    measurement_specs = {"action_noise_norm": SPEC_NOISE, "state_displacement_norm": SPEC_DISPLACEMENT,
                         "solver_residual": SPEC_SOLVER, "draw_index": SPEC_DRAW}

    def __init__(self, labels_per_transition: int, action_noise_std: float, max_iterations: int = 50,
                 tolerance: float = 1e-4, action_bounds=None):
        self.params = {**_noise_params(labels_per_transition, action_noise_std, action_bounds),
                       "max_iterations": int(max_iterations), "tolerance": float(tolerance)}
        if action_bounds is not None:
            self.measurement_specs = {**type(self).measurement_specs, "attempt_index": SPEC_ATTEMPT}

    def generate(self, batch, model, seed):
        p = self.params
        source, draw, attempt, a_tilde, noise_norm = _noisy_actions(
            batch, seed, p["labels_per_transition"], p["action_noise_std"], _bounds(p.get("action_bounds")))
        target = batch.z_next[source]
        z, failed, resid = newton_preimage(model, target, a_tilde, batch.z[source], p["max_iterations"], p["tolerance"],
                                           out_dtype=batch.z.dtype)
        return CandidateBatch(source=source, z=z, a=a_tilde, target=target.copy(), failed=failed,
                              measurements=_noise_measurements(attempt, action_noise_norm=noise_norm, draw_index=draw,
                                                               state_displacement_norm=_displacement(z, batch.z[source]),
                                                               solver_residual=resid))
