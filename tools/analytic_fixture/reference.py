"""Positive control: closed-form candidates with the declared validity rules.

This is not production code. It shows the oracle passes on an implementation
that follows the declared rules, so a legacy failure is the implementation's.
"""
import numpy as np

from system import DoubleIntegrator


def run(config, model, trajectories):
    sys_ = DoubleIntegrator(config["h"])
    rng = np.random.default_rng(config["seed"])
    transitions = [{"s": t["observations"][i], "a": t["actions"][i], "s_next": t["observations"][i + 1]}
                   for t in trajectories for i in range(len(t["observations"]) - 1)]
    results = {}
    for method, params in config["methods"].items():
        eps, n = params.get("epsilon"), params.get("num_labels", 1)
        out = []
        for tr in transitions:
            s_star, a_star, s_next = (np.asarray(tr[k], dtype=np.float64) for k in ("s", "a", "s_next"))
            if method == "noisy_action":
                a_c = a_star[None] + rng.normal(0.0, params["delta"], size=(n, 1))
                s_c, target = sys_.noisy_action(np.repeat(s_next[None], n, 0), a_c), s_next
            elif method == "forward_euler":
                a_c, s_c, target = a_star[None], sys_.exact_preimage(s_star[None], a_star[None]), s_star
            else:
                a_c, s_c, target = a_star[None], sys_.backward_euler(s_star[None], a_star[None]), s_star
            cands, counts = [], {"accepted": 0, "rejected": 0, "generation_failed": 0}
            for k in range(n):
                s_g, a_g = s_c[k].astype(np.float32), a_c[k].astype(np.float32)
                if not (np.all(np.isfinite(s_g)) and np.all(np.isfinite(a_g))):
                    ok, reason = False, "non_finite"
                elif eps is not None and np.linalg.norm(s_c[k] - s_star) > eps:
                    ok, reason = False, "out_of_support"
                else:
                    ok, reason = True, None
                counts["accepted" if ok else "rejected"] += 1
                cands.append({"s_G": s_g, "a_G": a_g, "target": target.astype(np.float32), "accepted": ok, "reason": reason})
            out.append({"attempted": n, "reported": counts, "candidates": cands})
        results[method] = out
    return transitions, results, {}
