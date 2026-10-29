#!/usr/bin/env python3
"""Historical Parts A and B of docs/experiments/pendulum-eval-v2-predeclaration.md (revision 2).

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \\
        python tools/pendulum_eval/historical.py part-a --root V6_ROOT
    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \\
        python tools/pendulum_eval/historical.py part-b --root V6_ROOT --out OUT_DIR [--device cuda]

Part A (post hoc, v6 rollouts, not confirmatory): every v6 study record's rollouts, sha256-checked against the record,
scored under pendulum_swingup_v2; paired union intervals at 95%, descriptive. Nothing is rerun.

Part B (predeclared fresh-start validation of CCIL versus BC): the frozen v6 checkpoints of the reused arms, each
checked against its record's sha256, evaluated on fresh starts SeedSequence([s + 1000, 7]) through the batched
evaluator. Before any fresh-start rollout, the loop and loader must reproduce stored v6 rollouts bit for bit. Noise
is reported per channel (observation only, action only) at v6's levels, with draws SeedSequence([s + 1000, 8, k]).
Primary: CCIL - BC in upright occupancy, the v6 decision rule at c = 0.975 with the BC - BC 10% control.

Reports (no machine paths) go to docs/fixtures/evidence/pendulum-eval-v2/part-{a,b}.json; Part B's rollouts go to
OUT_DIR, which must not exist, and their sha256 are in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "tools" / "pendulum_eval", REPO / "tools" / "vector_env", REPO / "tools" / "m3_pendulum"):
    sys.path.insert(0, str(p))
import env as canonical  # noqa: E402  (tools/m3_pendulum)
import pendulum as vector  # noqa: E402  (tools/vector_env)
import profiles  # noqa: E402

from ccil.experiments import analysis  # noqa: E402

EVIDENCE = REPO / "docs" / "fixtures" / "evidence" / "pendulum-eval-v2"
TASKS = ("continuous", "discontinuous")
SEEDS = tuple(range(1201, 1221))
FRESH_OFFSET = 1000
METRICS = ("upright_occupancy", "normalized_height", "final_hold")
LEVELS = canonical.NOISE_LEVELS
CHANNELS = ("observation_only", "action_only")
REUSED = ("bc", "ccil", "ccil_filtered", "ccil_nolip", "bc_frac10", "bc_frac25", "bc_frac50", "ccil_frac10",
          "ccil_frac25", "ccil_frac50", "bc_modern", "ccil_modern")
NOISY = ("bc", "ccil", "bc_modern", "ccil_modern")
PRIMARY = analysis.Decision(treatment="ccil", baseline="bc", control=("bc", "bc_frac10"), confidence=0.975,
                            min_valid=16, max_excluded=2,
                            outcomes=("ccil_improves_on_bc", "ccil_is_worse_than_bc", "no_detectable_difference"))
DATA_SIZE = [(f"{m}", f"{m}_frac{f}") for m in ("bc", "ccil") for f in (10, 25, 50)] + \
            [(f"ccil_frac{f}", f"bc_frac{f}") for f in (10, 25, 50)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rollout(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "meta"}


def seed_scores(rollout: dict) -> dict:
    """Per-seed v2 values: pooled occupancy and height, final-hold rate, and the summary."""
    s = profiles.score_rollout(rollout)["pendulum_swingup_v2"]
    return {"upright_occupancy": s["upright_occupancy"], "normalized_height": s["mean_normalized_height"],
            "final_hold": s["final_two_second_hold"], "summary": s}


def records(root: Path) -> list[dict]:
    out = []
    for f in sorted((root / "results" / "study").glob("*/*/s*.json")):
        out.append(json.loads(f.read_text()))
    return out


def tables_from(scores: dict, status: dict) -> dict:
    """{metric: analysis tables} from scores[(task, arm, noise_key, seed)] = seed_scores(...)."""
    out = {}
    for m in METRICS:
        t = {"R": {}, "status": status}
        for (task, arm, n, s), v in scores.items():
            t["R"].setdefault(task, {}).setdefault(arm, {}).setdefault(n, {})[s] = v[m]
        out[m] = t
    return out


def describe(tables: dict, pairs, noises, c=0.95) -> dict:
    out = {}
    for task in TASKS:
        for a, b in pairs:
            for n in noises:
                if n in tables["upright_occupancy"]["R"][task].get(a, {}) and \
                        n in tables["upright_occupancy"]["R"][task].get(b, {}):
                    out[f"{task}/{a}-{b}/{n}"] = {m: analysis.compare(tables[m], task, a, b, c, n) for m in METRICS}
    return out


def arm_summaries(scores: dict) -> dict:
    """Per task, arm and condition: the mean over seeds of every per-seed summary value."""
    groups = {}
    for (task, arm, n, _), v in scores.items():
        groups.setdefault(f"{task}/{arm}/{n}", []).append(v["summary"])
    out = {}
    for k, rows in groups.items():
        num = [x for x in rows[0] if isinstance(rows[0][x], (int, float)) and not isinstance(rows[0][x], bool)]
        out[k] = {x: float(np.mean([r[x] for r in rows if r[x] is not None])) if any(r[x] is not None for r in rows)
                  else None for x in num}
        out[k]["seeds"] = len(rows)
    return out


# --- Part A ----------------------------------------------------------------------------------------------------------
def part_a(root: Path) -> dict:
    scores, status, rollouts = {}, {}, {}
    for r in records(root):
        task, arm, seed = r["task"], r["arm"], r["seed"]
        status.setdefault(task, {}).setdefault(arm, {})[seed] = r["status"]
        for n, ev in sorted(r["evaluations"].items()):
            path = root / ev["rollout"]
            digest = sha(path)
            if digest != ev["rollout_sha256"]:
                raise SystemExit(f"Part A refused: {ev['rollout']} does not match its record's sha256")
            scores[(task, arm, n, seed)] = seed_scores(load_rollout(path))
            rollouts[f"{task}/{arm}/s{seed}/{n}"] = digest
    tables = tables_from(scores, status)
    noises = sorted({n for (_, _, n, _) in scores})
    arms = sorted({a for (_, a, _, _) in scores})
    pairs = [(a, "bc") for a in arms if a != "bc"] + [
        ("ccil", "noise_bc"), ("ccil_tuned", "noise_bc"), ("ccil_filtered", "ccil"), ("ccil_nolip", "ccil"),
        ("ccil_modern", "bc_modern"), ("ccil_modern", "ccil"), *DATA_SIZE] + [(a, "zero") for a in arms if a != "zero"]
    return {"label": "post hoc, v6 rollouts, not confirmatory", "profile_version": profiles.VERSION,
            "rollouts_scored": len(rollouts), "rollout_sha256": rollouts,
            "per_seed": {f"{t}/{a}/{n}/s{s}": {m: v[m] for m in METRICS} for (t, a, n, s), v in scores.items()},
            "arm_summaries": arm_summaries(scores), "comparisons_95": describe(tables, pairs, noises)}


# --- Part B ----------------------------------------------------------------------------------------------------------
def channel_rollouts(items: list, task: str) -> list[dict]:
    """vector_env.pendulum.rollouts with a noise channel per item: (policy, eval_seed, level, channel), channel one of
    "none", "both", "observation_only", "action_only". The loop is rollouts()'s, step for step; "both" with a v6 seed
    must reproduce the stored v6 rollout bit for bit (checked before any fresh-start rollout)."""
    discontinuous = canonical.TASKS[task]
    E, T, K = canonical.EPISODES, canonical.HORIZON, len(items)
    obs_noise, act_noise = np.zeros((K, E, T, 3)), np.zeros((K, E, T, 1))
    level = np.zeros((K, 1, 1))
    on_obs, on_act = np.zeros(K, bool), np.zeros(K, bool)
    states = np.full((K, E, T + 1, 3), np.nan)
    for i, (_, seed, n, ch) in enumerate(items):
        if ch != "none":
            obs_noise[i], act_noise[i] = canonical.noise_draws(seed, LEVELS.index(n))
            level[i] = n
            on_obs[i], on_act[i] = ch in ("both", "observation_only"), ch in ("both", "action_only")
        states[i, :, 0] = canonical.initial_states(seed)
    observations = np.full((K, E, T, 3), np.nan)
    policy_actions, actions = np.full((K, E, T, 1), np.nan), np.full((K, E, T, 1), np.nan)
    rewards = np.full((K, E, T), np.nan)
    flipped = np.zeros((K, E, T), dtype=bool)
    alive = np.ones((K, E), dtype=bool)
    for t in range(T):
        obs = states[:, :, t].copy()
        obs[on_obs] = obs[on_obs] * (1 - level[on_obs]) + level[on_obs] * obs_noise[on_obs, :, t]
        observations[alive, t] = obs[alive]
        pa = np.full((K, E, 1), np.nan)
        for i, (policy, _, _, _) in enumerate(items):
            if alive[i].any():
                pa[i, alive[i]] = np.asarray(policy(obs[i, alive[i]]), dtype=np.float64).reshape(-1, 1)
        policy_actions[alive, t] = pa[alive]
        act = pa.copy()
        act[on_act] = pa[on_act] * (1 - level[on_act]) + level[on_act] * act_noise[on_act, :, t]
        alive &= np.isfinite(act[:, :, 0])
        if alive.any():
            actions[alive, t] = act[alive]
            rewards[alive, t] = vector.reward(states[alive, t], act[alive][:, 0])
            states[alive, t + 1], flipped[alive, t] = vector.step(states[alive, t], act[alive][:, 0], discontinuous)
    out = []
    for i, (_, seed, n, ch) in enumerate(items):
        returns = rewards[i].sum(axis=1)
        out.append({"states": states[i], "observations": observations[i], "policy_actions": policy_actions[i],
                    "actions": actions[i], "rewards": rewards[i], "flipped": flipped[i], "returns": returns,
                    "R": np.float64(returns.mean()), "failed_episodes": np.flatnonzero(~alive[i]),
                    "meta": {"version": "pendulum_eval.historical/1", "task": task, "eval_seed": seed,
                             "noise_level": n, "channel": ch, "integrator": "batched RK45 (numpy)"}})
    return out


def _zero(obs):
    return np.zeros((len(obs), 1))


def load_policies(root: Path, device: str) -> dict:
    from ccil.training.vector_bc import VectorBCPolicy
    pol = {}
    for r in records(root):
        if r["arm"] not in REUSED:
            continue
        ckpt = Path(r["stages"]["policy.train"]) / "checkpoints" / "policy.npz"
        if sha(ckpt) != r["checkpoint_sha256"]:
            raise SystemExit(f"Part B refused: {r['task']}/{r['arm']}/s{r['seed']} checkpoint does not match its record")
        pol[(r["task"], r["arm"], r["seed"])] = (VectorBCPolicy.load(ckpt, device=device)[0].predict, r)
    return pol


def reproduce_v6(root: Path, pol: dict) -> dict:
    """The loop and loader reproduce stored v6 rollouts bit for bit: bc and ccil, two seeds, noise-free and 0.1."""
    checked = []
    for task in TASKS:
        items, refs = [], []
        for arm in ("bc", "ccil"):
            for s in SEEDS[:2]:
                predict, r = pol[(task, arm, s)]
                for n, ch in ((0.0, "none"), (0.1, "both")):
                    items.append((predict, s, n, ch))
                    refs.append((f"{task}/{arm}/s{s}/{n}", root / r["evaluations"][str(n)]["rollout"]))
        for res, (name, path) in zip(channel_rollouts(items, task), refs):
            stored = load_rollout(path)
            same = all(np.array_equal(res[k], stored[k], equal_nan=True)
                       for k in ("states", "observations", "policy_actions", "actions"))
            checked.append({"rollout": name, "bitwise_equal": bool(same)})
    if not all(c["bitwise_equal"] for c in checked):
        raise SystemExit(f"Part B refused before any fresh-start rollout: v6 not reproduced: {checked}")
    return {"reproduced": checked}


def part_b(root: Path, out: Path, device: str) -> dict:
    out.mkdir(parents=True)  # refuses an existing directory: rollouts are never overwritten
    pol = load_policies(root, device)
    report = {"label": "predeclared fresh-start validation of CCIL versus BC (historical demonstrator)",
              "profile_version": profiles.VERSION, "device": device, "fresh_offset": FRESH_OFFSET,
              "loop_check": reproduce_v6(root, pol)}
    scores, status, rollouts, noisy_failures = {}, {}, {}, []
    for task in TASKS:
        items, keys = [], []
        for arm in (*REUSED, "zero"):
            for s in SEEDS:
                predict = _zero if arm == "zero" else pol[(task, arm, s)][0]
                conds = [("0.0", 0.0, "none")]
                if arm in NOISY:
                    conds += [(f"{ch}/{n}", n, ch) for ch in CHANNELS for n in LEVELS]
                for key, n, ch in conds:
                    items.append((predict, s + FRESH_OFFSET, n, ch))
                    keys.append((task, arm, key, s))
        for res, (t, arm, key, s) in zip(channel_rollouts(items, task), keys):
            path = out / t / arm / f"s{s}-{key.replace('/', '-')}.npz"
            canonical.save(res, path)
            rollouts[f"{t}/{arm}/s{s}/{key}"] = sha(path)
            st = status.setdefault(t, {}).setdefault(arm, {})
            if len(res["failed_episodes"]):  # non-finite actions: an algorithmic failure, never scored
                if key == "0.0":
                    st[s] = "algorithmic_failure"
                noisy_failures.append(f"{t}/{arm}/s{s}/{key}")
            else:
                if key == "0.0":
                    st[s] = "ok"
                scores[(t, arm, key, s)] = seed_scores(res)
        print(task, "evaluated", len(items), "rollouts", flush=True)
    tables = tables_from(scores, status)
    report["primary"] = {t: analysis.decide(tables["upright_occupancy"], t, PRIMARY) for t in TASKS}
    report["failures"] = {t: {a: analysis.counts(tables["upright_occupancy"], t, a) for a in status[t]} for t in TASKS}
    report["failed_rollouts"] = noisy_failures  # every rollout with non-finite actions, noise-free or noisy
    pairs = [(a, "bc") for a in (*REUSED, "zero") if a != "bc"] + [("ccil_modern", "bc_modern"), *DATA_SIZE]
    report["secondary_95"] = describe(tables, pairs, ["0.0"])
    report["noise_95"] = describe(tables, [("ccil", "bc"), ("ccil_modern", "bc_modern")],
                                  [f"{ch}/{n}" for ch in CHANNELS for n in LEVELS])
    report["arm_summaries"] = arm_summaries(scores)
    report["per_seed"] = {f"{t}/{a}/{n}/s{s}": {m: v[m] for m in METRICS} for (t, a, n, s), v in scores.items()}
    report["rollout_sha256"] = rollouts
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("part-a", "part-b"))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.part == "part-a":
        report, dest = part_a(args.root), EVIDENCE / "part-a.json"
    else:
        if args.out is None:
            ap.error("part-b needs --out")
        report, dest = part_b(args.root, args.out, args.device), EVIDENCE / "part-b.json"
    dest.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
