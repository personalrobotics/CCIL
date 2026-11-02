"""The Gate 3A funnel: accepted training rows surviving each supported-row condition in turn, per r and sigma.

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt python docs/fixtures/evidence/pendulum-correction/funnel.py SCAN_DIR
"""
import json, sys
from pathlib import Path
import numpy as np
D = Path(sys.argv[1]); doc = json.loads((D / "correction.json").read_text())
print("tau", doc["tau_demo"])
for r, sp in doc["splits"].items():
    print(f"--- r = {r} (needed >= 100 supported, near-identity <= 0.5)")
    print("sigma   acc  nearId  moved  onM  rc<=1e-3 |da-piD|<=.3 notSw  dDemo>=tau  outLQR recov | d_source q50/q95  dDemo q95  consist q50")
    for s in doc["sigma_grid"]:
        f = np.load(D / f"rows-a-sigma{s:g}.npz")
        tr = np.isin(f["episode"], sp["train"]); m = tr & f["accepted"]; n = [int(m.sum())]
        near = np.isin(f["class"], (0, 1))
        c = doc["cells"][r][f"{s:g}"]["a"]
        m = m & ~near; n.append(int(m.sum()))
        for cond in (f["manifold_error"] <= 6.7e-7, f["construction_residual"] <= 1e-3, f["consistency"] <= 0.3,
                     f["switch_margin"] >= 0.01, f[f"d_demo_r{float(r):g}"] >= doc["tau_demo"][r], f["region"] > 0, f["recovered"] == 1):
            m = m & cond; n.append(int(m.sum()))
        acc = tr & f["accepted"]
        ds, dd, co = f["d_source"][acc], f[f"d_demo_r{float(r):g}"][acc], f["consistency"][acc]
        print(f"{s:<7g} {n[0]:>6} {c['near_identity_fraction']:.2f} " + " ".join(f"{x:>6}" for x in n[1:]),
              f"| {np.quantile(ds,.5):.2e}/{np.quantile(ds,.95):.2e}  {np.quantile(dd,.95):.3f}  {np.quantile(co,.5):.2f}  supp={c['supported']} pass={c['pass']}")
