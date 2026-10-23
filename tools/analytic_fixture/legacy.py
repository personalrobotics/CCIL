"""Run the archived legacy generators against the analytic fixture.

Functions are loaded verbatim from the archive tag with `ast`, so the legacy
module graph (gym, d4rl, mujoco-py) is never imported.
"""
import ast
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

TAG = "archive/upstream-main-2026-10-22"
SOURCES = {
    "correct_il/gen_aug_label.py": ["gen_forward_euler", "gen_backward_euler", "gen_backward_euler_fast", "gen_noisy_action_be"],
    "correct_il/utils.py": ["root_finder_s", "build_dataset"],
}
REPO = Path(__file__).resolve().parents[2]


def load_legacy():
    ns = {"torch": torch, "np": np, "Tuple": Tuple}
    for path, names in SOURCES.items():
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{TAG}:{path}"],
                             capture_output=True, text=True, check=True).stdout
        tree = ast.parse(src)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        missing = set(names) - {n.name for n in nodes}
        if missing:
            raise RuntimeError(f"{TAG}:{path} lacks {sorted(missing)}")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{TAG}:{path}", "exec"), ns)
    ns["commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{TAG}^{{commit}}"],
                                  capture_output=True, text=True, check=True).stdout.strip()
    return ns


def run(config, model, trajectories):
    legacy = load_legacy()
    np_s, np_a, np_sp = legacy["build_dataset"](trajectories)
    transitions = [{"s": s, "a": a, "s_next": sp} for s, a, sp in zip(np_s, np_a, np_sp)]
    s, a, sp = (torch.as_tensor(x).float() for x in (np_s, np_a, np_sp))

    generators = {
        "backward_euler_fast": lambda p: (lambda m, d: legacy["gen_backward_euler_fast"](p["max_iter"], m, d)),
        "backward_euler": lambda p: (lambda m, d: legacy["gen_backward_euler"](p["max_iter"], m, d)),
        "forward_euler": lambda p: (lambda m, d: legacy["gen_forward_euler"](p["max_iter"], m, d)),
        "noisy_action": lambda p: (lambda m, d: legacy["gen_noisy_action_be"](p["num_labels"], p["delta"], p["epsilon"], m, d)),
    }
    results = {}
    for method, params in config["methods"].items():
        torch.manual_seed(config["seed"])  # legacy main() calls seed() once before generating
        gen = generators[method](params)
        out = []
        for i in range(len(np_s)):
            n_suc, n_fail, gen_data, _ = gen(model, (s[i], a[i], sp[i]))
            # Mirrors legacy main() (gen_aug_label.py:250-260): every row of gen_data is
            # written to aug_data.pkl whenever the generator reports _count_suc > 0.
            accepted = n_suc > 0
            rows = [] if gen_data[0] is None else [np.atleast_2d(np.asarray(x, dtype=np.float32)) for x in gen_data]
            cands = []
            if rows:
                for k in range(rows[0].shape[0]):
                    cands.append({"s_G": rows[0][k], "a_G": rows[1][k], "target": rows[2][k],
                                  "accepted": bool(accepted), "reason": None if accepted else "legacy_failure"})
            out.append({"attempted": params.get("num_labels", 1),
                        "reported": {"accepted": int(n_suc), "rejected": int(n_fail), "generation_failed": 0},
                        "candidates": cands})
        results[method] = out
    return transitions, results, {"legacy_commit": legacy["commit"]}
