"""Version-3 (698262f) and version-5 improper sets on the smoke grids, with the version-5 verifier.

    git show 698262f:tools/pendulum_oracle/oracle.py > /tmp/oracle_old.py
    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \
        python docs/fixtures/evidence/pendulum-oracle/improper_equivalence.py /tmp/oracle_old.py
"""
import importlib.util, json, sys, time
sys.path.insert(0, "tools/pendulum_oracle")
import numpy as np, oracle as O
spec = importlib.util.spec_from_file_location("oracle_old", sys.argv[1]); old = importlib.util.module_from_spec(spec)
sys.modules["oracle_old"] = old; spec.loader.exec_module(old)
spec = importlib.util.spec_from_file_location("oracle_check", "tools/pendulum_oracle/check.py"); C = importlib.util.module_from_spec(spec)
sys.modules["oracle_check"] = C; spec.loader.exec_module(C)
SMOKE = {"S1": (40, 49, 7), "S2": (80, 97, 13), "S3": (160, 193, 25)}
O.LEVELS.update(SMOKE)
A, B = O.linearize(); Ls = {b: O.lqr_bundle(b, A, B) for b in C.ALL_BETAS}
rows = []
for name in SMOKE:
    for wmax in (O.OMEGA_MAX, O.OMEGA_MAX_CHECK):
        g = O.Grid.level(name, wmax=wmax); T = O.transitions(g, 8)
        for b in C.ALL_BETAS:
            t = time.time(); ro = old.improper(g, T, Ls[b]); to = time.time() - t
            t = time.time(); rn = O.improper(g, T, Ls[b]); tn = time.time() - t
            v = C.verify_improper(g, T, Ls[b], rn)
            same = {"eq_" + k: bool((ro[k] == rn[k]).all()) for k in ("improper", "self_loop", "no_live_action")}
            rows.append(dict(grid=name, wmax=wmax, nodes=len(g.E), actions=len(g.u), beta=b, improper=int(rn["improper"].sum()),
                             old_rounds=ro["rounds"], old_s=round(to, 2), new_s=round(tn, 3), **same, verify=v))
            print(json.dumps(rows[-1]), flush=True)
print("ALL EQUAL AND VERIFIED", all(r["eq_improper"] and r["eq_self_loop"] and r["eq_no_live_action"] and r["verify"]["pass"] for r in rows), len(rows), "cases")
