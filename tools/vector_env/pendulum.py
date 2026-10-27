"""Batched Pendulum dynamics, numerically equivalent to the canonical environment (#111).

The canonical environment (tools/m3_pendulum/env.py) advances one episode at a
time: each step is scipy.integrate.solve_ivp(RK45, (0, dt), default tolerances,
dense_output=True) evaluated at dt. step() here advances N states at once by
the same algorithm, vectorized, with every episode keeping its own adaptive
step sequence (scipy 1.16.3's RK45: select_initial_step, the Dormand-Prince
5(4) tableau, the error norm and step-size control with SAFETY 0.9 and
factors in [0.2, 10], and the RkDenseOutput evaluation at dt). The arithmetic
is the same per episode up to the summation order of a few dot products, so
results agree with the canonical environment to a few ulps; the evidence
reports the measured maximum and checks every stored demonstration transition.

reward() and the discontinuity are the canonical environment's.
"""
from __future__ import annotations

import numpy as np

G, LENGTH, DT = 9.81, 1.0, 0.02
RTOL, ATOL = 1e-3, 1e-6
SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.2, 10.0
ORDER, ERROR_ORDER = 5, 4  # RK45: the error estimator has order 4
C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
A = np.array([[0, 0, 0, 0, 0],
              [1 / 5, 0, 0, 0, 0],
              [3 / 40, 9 / 40, 0, 0, 0],
              [44 / 45, -56 / 15, 32 / 9, 0, 0],
              [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
              [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]])
B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
E = np.array([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40])
P = np.array([[1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
              [0, 0, 0, 0],
              [0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
              [0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
              [0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
              [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
              [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423]])
N_STAGES = 6


def field(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The canonical vector field on (N, 3) states and (N,) actions."""
    return np.stack([y[:, 2] * y[:, 1], y[:, 2] * -y[:, 0], -(G / LENGTH) * y[:, 0] + a], axis=1)


def _norm(x: np.ndarray) -> np.ndarray:
    """scipy's RMS norm, per row."""
    return np.sqrt((x * x).sum(axis=1)) / np.sqrt(x.shape[1])


def _initial_step(y0, f0, a, t_bound):
    scale = ATOL + np.abs(y0) * RTOL
    d0, d1 = _norm(y0 / scale), _norm(f0 / scale)
    h0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / np.where(d1 == 0, 1, d1))
    h0 = np.minimum(h0, t_bound)
    f1 = field(y0 + h0[:, None] * f0, a)
    d2 = _norm((f1 - f0) / scale) / h0
    tiny = (d1 <= 1e-15) & (d2 <= 1e-15)
    h1 = np.where(tiny, np.maximum(1e-6, h0 * 1e-3),
                  (0.01 / np.where(tiny, 1, np.maximum(d1, d2))) ** (1 / (ERROR_ORDER + 1)))
    return np.minimum(np.minimum(100 * h0, h1), t_bound)


def integrate(y0: np.ndarray, a: np.ndarray, t_bound: float = DT) -> np.ndarray:
    """solve_ivp(RK45, (0, t_bound)).sol(t_bound) for every row of y0 (N, 3) under actions a (N,)."""
    y0, a = np.asarray(y0, np.float64), np.asarray(a, np.float64)
    n = len(y0)
    t = np.zeros(n)
    y = y0.copy()
    f = field(y, a)
    h_abs = _initial_step(y, f, a, t_bound)
    done = np.zeros(n, bool)
    y_old, t_old, h_last = y.copy(), np.zeros(n), np.ones(n)
    Q = np.zeros((n, 3, 4))
    K = np.zeros((N_STAGES + 1, n, 3))
    exponent = -1 / (ERROR_ORDER + 1)
    while not done.all():
        act = np.flatnonzero(~done)
        ta, ya, fa, ha, aa = t[act], y[act], f[act], h_abs[act], a[act]
        min_step = 10 * np.abs(np.nextafter(ta, np.inf) - ta)
        ha = np.clip(ha, min_step, np.inf)
        rejected = np.zeros(len(act), bool)
        accepted = np.zeros(len(act), bool)
        out_y, out_f, out_t, out_h, out_k = (np.empty_like(ya), np.empty_like(fa), np.empty_like(ta), np.empty_like(ta),
                                             np.empty((N_STAGES + 1, len(act), 3)))
        trial = np.arange(len(act))  # rows of act still searching for an accepted step
        while len(trial):
            if (ha[trial] < min_step[trial]).any():
                raise FloatingPointError("RK45 step size fell below the minimum (scipy: TOO_SMALL_STEP)")
            tt, yy, ff, aa_ = ta[trial], ya[trial], fa[trial], aa[trial]
            t_new = np.minimum(tt + ha[trial], t_bound)
            h = t_new - tt
            k = np.empty((N_STAGES + 1, len(trial), 3))
            k[0] = ff
            for s in range(1, N_STAGES):
                dy = np.einsum("s,snd->nd", A[s, :s], k[:s]) * h[:, None]
                k[s] = field(yy + dy, aa_)
            y_new = yy + h[:, None] * np.einsum("s,snd->nd", B, k[:-1])
            f_new = field(y_new, aa_)
            k[-1] = f_new
            scale = ATOL + np.maximum(np.abs(yy), np.abs(y_new)) * RTOL
            err = _norm(np.einsum("s,snd->nd", E, k) * h[:, None] / scale)
            ok = err < 1
            with np.errstate(divide="ignore"):  # err == 0 gives inf, used by neither: grow takes MAX_FACTOR there and
                grow = np.where(err == 0, MAX_FACTOR, np.minimum(MAX_FACTOR, SAFETY * err ** exponent))
                shrink = np.maximum(MIN_FACTOR, SAFETY * err ** exponent)  # shrink applies only where err >= 1
            grow = np.where(rejected[trial], np.minimum(1, grow), grow)
            ha[trial] = np.abs(h) * np.where(ok, grow, shrink)
            acc = trial[ok]
            out_y[acc], out_f[acc], out_t[acc], out_h[acc] = y_new[ok], f_new[ok], t_new[ok], h[ok]
            out_k[:, acc] = k[:, ok]
            accepted[acc] = True
            rejected[trial[~ok]] = True
            trial = trial[~ok]
        y_old[act], t_old[act], h_last[act] = ya, ta, out_h
        Q[act] = np.einsum("snd,sj->ndj", out_k, P)
        t[act], y[act], f[act], h_abs[act] = out_t, out_y, out_f, ha
        done[act] = out_t >= t_bound
    x = (t_bound - t_old) / h_last
    p = np.cumprod(np.repeat(x[:, None], 4, axis=1), axis=1)
    return h_last[:, None] * np.einsum("ndj,nj->nd", Q, p) + y_old


def step(states: np.ndarray, actions: np.ndarray, discontinuous: bool):
    """One environment step for N states: (next states (N, 3), flipped (N,))."""
    nxt = integrate(states, actions)
    flipped = np.zeros(len(nxt), bool)
    if discontinuous:
        theta = np.arctan2(nxt[:, 0], nxt[:, 1])
        flipped = (np.abs(theta - np.pi / 6) < 1e-2) | (np.abs(theta - np.pi / 3) < 1e-2)
        nxt[flipped, 2] = -nxt[flipped, 2]
    return nxt, flipped


def reward(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """The canonical reward on the pre-step states and applied actions, per row."""
    theta = np.arctan2(states[:, 0], states[:, 1])
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    d = np.stack([theta - np.pi, states[:, 2]], axis=1)
    return -0.5 * np.sqrt((d * d).sum(axis=1)) ** 2 - 0.5 * actions ** 2  # the canonical norm, then squared


def rollout(policy, task: str, seed: int, noise_level: float = 0.0) -> dict:
    """tools/m3_pendulum/env.rollout with every episode advanced together by step().

    The same initial states, noise draws, failure handling and returned arrays, so audit.py applies unchanged; the
    meta names this implementation.
    """
    import env as canonical  # tools/m3_pendulum: initial states, noise draws, constants
    discontinuous = canonical.TASKS[task]
    E, T = canonical.EPISODES, canonical.HORIZON
    if noise_level:
        k = canonical.NOISE_LEVELS.index(noise_level)
        obs_noise, act_noise = canonical.noise_draws(seed, k)
    else:
        k, obs_noise, act_noise = None, None, None
    states = np.full((E, T + 1, 3), np.nan)
    states[:, 0] = canonical.initial_states(seed)
    observations = np.full((E, T, 3), np.nan)
    policy_actions = np.full((E, T, 1), np.nan)
    actions = np.full((E, T, 1), np.nan)
    rewards = np.full((E, T), np.nan)
    flipped = np.zeros((E, T), dtype=bool)
    alive = np.ones(E, dtype=bool)
    for t in range(T):
        obs = states[:, t]
        if noise_level:
            obs = obs * (1 - noise_level) + noise_level * obs_noise[:, t]
        observations[alive, t] = obs[alive]
        pa = np.full((E, 1), np.nan)
        pa[alive] = np.asarray(policy(obs[alive]), dtype=np.float64).reshape(-1, 1)
        policy_actions[alive, t] = pa[alive]
        act = pa * (1 - noise_level) + noise_level * act_noise[:, t] if noise_level else pa
        alive &= np.isfinite(act[:, 0])
        live = np.flatnonzero(alive)
        if len(live):
            actions[live, t] = act[live]
            rewards[live, t] = reward(states[live, t], act[live, 0])
            states[live, t + 1], flipped[live, t] = step(states[live, t], act[live, 0], discontinuous)
    returns = rewards.sum(axis=1)
    return {"states": states, "observations": observations, "policy_actions": policy_actions, "actions": actions,
            "rewards": rewards, "flipped": flipped, "initial_states": states[:, 0].copy(), "returns": returns,
            "R": np.float64(returns.mean()), "failed_episodes": np.flatnonzero(~alive),
            **({"obs_noise": obs_noise, "act_noise": act_noise} if noise_level else {}),
            "meta": {"version": "vector_env.pendulum/1", "task": task, "seed": seed, "noise_level": noise_level,
                     "noise_index": k, "episodes": E, "horizon": T, "integrator": "batched RK45 (numpy)"}}


def rollouts(items: list, task: str) -> list[dict]:
    """rollout() for many (policy, seed, noise_level) items at once: one step() over every live episode of every
    item; each item's policy is called on exactly the rows rollout() would give it (its live episodes, in order).

    The dynamics are row-wise and the policy inputs are rollout()'s, so each result equals rollout(policy, task,
    seed, noise_level) bit for bit, whatever else is in the batch (the evidence compares the two). Calling a
    policy once on several items' rows would change its float32 batch size and, through rounding, its actions.
    """
    import env as canonical
    discontinuous = canonical.TASKS[task]
    E, T, K = canonical.EPISODES, canonical.HORIZON, len(items)
    obs_noise = np.zeros((K, E, T, 3))
    act_noise = np.zeros((K, E, T, 1))
    level = np.zeros((K, 1, 1))
    ks = []
    states = np.full((K, E, T + 1, 3), np.nan)
    for i, (_, seed, n) in enumerate(items):
        k = canonical.NOISE_LEVELS.index(n) if n else None
        ks.append(k)
        if n:
            obs_noise[i], act_noise[i] = canonical.noise_draws(seed, k)
            level[i] = n
        states[i, :, 0] = canonical.initial_states(seed)
    observations = np.full((K, E, T, 3), np.nan)
    policy_actions = np.full((K, E, T, 1), np.nan)
    actions = np.full((K, E, T, 1), np.nan)
    rewards = np.full((K, E, T), np.nan)
    flipped = np.zeros((K, E, T), dtype=bool)
    alive = np.ones((K, E), dtype=bool)
    noisy = np.array([bool(n) for _, _, n in items])
    for t in range(T):
        obs = states[:, :, t].copy()
        obs[noisy] = obs[noisy] * (1 - level[noisy]) + level[noisy] * obs_noise[noisy, :, t]
        observations[alive, t] = obs[alive]
        pa = np.full((K, E, 1), np.nan)
        for i, (policy, _, _) in enumerate(items):
            if alive[i].any():
                pa[i, alive[i]] = np.asarray(policy(obs[i, alive[i]]), dtype=np.float64).reshape(-1, 1)
        policy_actions[alive, t] = pa[alive]
        act = pa.copy()
        act[noisy] = pa[noisy] * (1 - level[noisy]) + level[noisy] * act_noise[noisy, :, t]
        alive &= np.isfinite(act[:, :, 0])
        if alive.any():
            actions[alive, t] = act[alive]
            rewards[alive, t] = reward(states[alive, t], act[alive][:, 0])
            states[alive, t + 1], flipped[alive, t] = step(states[alive, t], act[alive][:, 0], discontinuous)
    out = []
    for i, (_, seed, n) in enumerate(items):
        returns = rewards[i].sum(axis=1)
        out.append({"states": states[i], "observations": observations[i], "policy_actions": policy_actions[i],
                    "actions": actions[i], "rewards": rewards[i], "flipped": flipped[i],
                    "initial_states": states[i, :, 0].copy(), "returns": returns, "R": np.float64(returns.mean()),
                    "failed_episodes": np.flatnonzero(~alive[i]),
                    **({"obs_noise": obs_noise[i], "act_noise": act_noise[i]} if n else {}),
                    "meta": {"version": "vector_env.pendulum/1", "task": task, "seed": seed, "noise_level": n,
                             "noise_index": ks[i], "episodes": E, "horizon": T, "integrator": "batched RK45 (numpy)"}})
    return out


def replay_mismatches(rollout_arrays: dict, task: str) -> int:
    """Transitions of a saved rollout that step() does not reproduce bit for bit (the audit's replay, batched)."""
    import env as canonical
    s, a, fl = rollout_arrays["states"], rollout_arrays["actions"], rollout_arrays["flipped"]
    bad = 0
    for t in range(s.shape[1] - 1):
        live = np.isfinite(a[:, t, 0])
        if live.any():
            nxt, f = step(s[live, t], a[live, t, 0], canonical.TASKS[task])
            bad += int((~(nxt == s[live, t + 1]).all(axis=1) | (f != fl[live, t])).sum())
    return bad


def vector_env(task: str, num_envs: int = 100):
    """A gymnasium.vector.VectorEnv for the canonical Pendulum (autoreset disabled; truncation at the horizon).

    reset(seed=s) places episode i at the canonical initial state i of seed s; actions are applied unclipped, as
    the data generator did (the action space records the legacy bounds).
    """
    import gymnasium
    from gymnasium.vector import AutoresetMode, VectorEnv

    import env as canonical

    class PendulumVectorEnv(VectorEnv):
        metadata = {"autoreset_mode": AutoresetMode.DISABLED}

        def __init__(self):
            self.num_envs = num_envs
            self.single_observation_space = gymnasium.spaces.Box(canonical.OBS_LOW, canonical.OBS_HIGH, dtype=np.float64)
            self.single_action_space = gymnasium.spaces.Box(canonical.ACT_LOW, canonical.ACT_HIGH, dtype=np.float64)
            self.observation_space = gymnasium.vector.utils.batch_space(self.single_observation_space, num_envs)
            self.action_space = gymnasium.vector.utils.batch_space(self.single_action_space, num_envs)
            self.task, self.state, self.t = task, None, 0

        def reset(self, *, seed=None, options=None):
            if seed is None:
                raise ValueError("the canonical Pendulum needs a seed: initial states come from SeedSequence([seed, 7])")
            if num_envs > canonical.EPISODES:
                raise ValueError(f"a seed defines {canonical.EPISODES} initial states")
            self.state, self.t = canonical.initial_states(seed)[:num_envs].copy(), 0
            return self.state.copy(), {}

        def step(self, actions):
            a = np.asarray(actions, np.float64).reshape(num_envs)
            r = reward(self.state, a)
            self.state, flipped = step(self.state, a, canonical.TASKS[self.task])
            self.t += 1
            truncated = np.full(num_envs, self.t >= canonical.HORIZON)
            return self.state.copy(), r, np.zeros(num_envs, bool), truncated, {"flipped": flipped}

    return PendulumVectorEnv()
