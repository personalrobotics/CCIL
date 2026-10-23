"""Ground truth and invariant checks for the analytic fixture.

The oracle derives everything from the config and the closed forms in
system.py. It never calls an implementation; it only inspects the
transitions and candidate records an implementation returns.
"""
import math

import numpy as np

from system import DoubleIntegrator


def episodes_to_transitions(config):
    """Expected transitions (float32, as stored), one list per episode."""
    sys_ = DoubleIntegrator(config["h"])
    out = []
    for ep in config["episodes"]:
        s = np.array(ep["s0"], dtype=np.float64)
        states = [s]
        for a in ep["actions"]:
            s = sys_.step_np(s[None], np.array([[a]]))[0]
            states.append(s)
        actions = np.array(ep["actions"], dtype=np.float64)
        actions[list(ep["corrupt_actions"])] = np.nan  # corrupted after rollout: states stay consistent
        for t in range(len(actions)):
            out.append({
                "episode": ep["id"], "t": t,
                "s": states[t].astype(np.float32), "a": actions[t:t + 1].astype(np.float32),
                "s_next": states[t + 1].astype(np.float32),
            })
    return out


def _finite(x):
    return x is not None and bool(np.all(np.isfinite(np.asarray(x, dtype=np.float64))))


def _same(x, y, tol):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return x.shape == y.shape and bool(np.all((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= tol)))


def evaluate(config, transitions, results):
    """Return (annotated results, verdicts) for an implementation's output.

    transitions: list of {"s", "a", "s_next"} in the implementation's order.
    results: {method: [ {attempted, reported{accepted,rejected,generation_failed}, candidates[...]} per transition ]}
    """
    sys_ = DoubleIntegrator(config["h"])
    tol, band = config["tolerance"]["abs"], config["tolerance"]["boundary_rel"]
    expected = episodes_to_transitions(config)
    verdicts = {}

    i6 = len(transitions) == len(expected) and all(
        _same(t["s"], e["s"], 0) and _same(t["a"], e["a"], 0) and _same(t["s_next"], e["s_next"], 0)
        for t, e in zip(transitions, expected))
    verdicts["I6_episode_boundaries"] = {"pass": i6, "expected_transitions": len(expected), "got": len(transitions)}
    if not i6:
        return results, verdicts

    for method, params in config["methods"].items():
        eps, n_expected = params.get("epsilon"), params.get("num_labels", 1)
        viol = {k: [] for k in ("I1_accepted_finite", "I2_accepted_in_support", "I3_accounting",
                                "I4_closed_form", "I5_expected_acceptance")}
        for i, (dp, tr) in enumerate(zip(results[method], expected)):
            where = f"{tr['episode']}[{tr['t']}]"
            s_star, a_star, s_next = (np.asarray(tr[k], dtype=np.float64) for k in ("s", "a", "s_next"))
            cands, rep = dp["candidates"], dp["reported"]
            n_acc = sum(c["accepted"] for c in cands)
            if not (dp["attempted"] == n_expected and len(cands) == n_expected
                    and rep["accepted"] + rep["rejected"] + rep["generation_failed"] == n_expected
                    and rep["accepted"] == n_acc):
                viol["I3_accounting"].append(
                    f"{where}: attempted={dp['attempted']} records={len(cands)} reported={rep} accepted_records={n_acc} expected={n_expected}")
            for k, c in enumerate(cands):
                tag = f"{where}#{k}"
                a_g = None if c["a_G"] is None else np.asarray(c["a_G"], dtype=np.float64)
                if not _finite(a_star) or (method == "noisy_action" and not _finite(a_g)):
                    exp_class, s_exp, target, pred_res = "non_finite", None, None, None
                else:
                    if method == "noisy_action":
                        s_exp, target = sys_.noisy_action(s_next[None], a_g[None])[0], s_next
                        pred_res = sys_.h ** 2 * abs(float(a_g[0]))
                    elif method == "forward_euler":
                        s_exp, target, pred_res = sys_.exact_preimage(s_star[None], a_star[None])[0], s_star, 0.0
                    else:
                        s_exp, target = sys_.backward_euler(s_star[None], a_star[None])[0], s_star
                        pred_res = sys_.h ** 2 * abs(float(a_star[0]))
                    dist = float(np.linalg.norm(s_exp - s_star))
                    if eps is None or dist < eps * (1 - band):
                        exp_class = "accepted"
                    elif dist > eps * (1 + band):
                        exp_class = "out_of_support"
                    else:
                        exp_class = "boundary"
                    c["oracle_distance"] = dist
                c["expected_class"] = exp_class

                if c["accepted"] and not (_finite(c["s_G"]) and _finite(c["a_G"])):
                    viol["I1_accepted_finite"].append(f"{tag}: accepted non-finite label")
                if c["accepted"] and eps is not None and _finite(c["s_G"]):
                    d = float(np.linalg.norm(np.asarray(c["s_G"], dtype=np.float64) - s_star))
                    if d > eps * (1 + band):
                        viol["I2_accepted_in_support"].append(f"{tag}: accepted at distance {d:.4g} > epsilon {eps}")
                if s_exp is not None and c["s_G"] is not None and _finite(c["s_G"]):
                    meas = float(sys_.residual(np.asarray(c["s_G"], dtype=np.float64), a_g if a_g is not None else a_star, target))
                    c["oracle_residual"], c["predicted_residual"] = meas, pred_res
                    if not (_same(c["s_G"], s_exp, tol) and _same(c["target"], target, tol) and abs(meas - pred_res) <= tol):
                        viol["I4_closed_form"].append(f"{tag}: s_G={c['s_G']} expected={s_exp.tolist()} residual={meas:.3g} predicted={pred_res:.3g}")
                if exp_class != "boundary" and c["accepted"] != (exp_class == "accepted"):
                    viol["I5_expected_acceptance"].append(f"{tag}: accepted={c['accepted']} expected={exp_class}")
        for name, v in viol.items():
            verdicts.setdefault(name, {})[method] = {"pass": not v, "violations": len(v), "examples": v[:5]}
    return results, verdicts


def all_pass(verdicts):
    def ok(v):
        return v["pass"] if "pass" in v else all(ok(x) for x in v.values())
    return all(ok(v) for v in verdicts.values())


def jsonable(x):
    """Deterministic JSON encoding; non-finite floats become strings."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else str(x)
    if isinstance(x, np.integer):
        return int(x)
    return x
