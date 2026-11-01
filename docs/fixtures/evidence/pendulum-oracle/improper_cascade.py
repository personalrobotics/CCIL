"""Removal-batch map of the L1 improper certificate at beta = 1e-2.

    uv run --group evidence --with-requirements tools/vector_env/requirements.txt \
        python docs/fixtures/evidence/pendulum-oracle/improper_cascade.py <run>/improper-L1.npz OUT.svg
"""
import sys; sys.path.insert(0, "tools/pendulum_oracle")
import numpy as np, oracle as O, matplotlib; matplotlib.use("svg"); import matplotlib.pyplot as plt
plt.rcParams.update({"svg.hashsalt": "pendulum-oracle", "svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 8})
c = np.load(sys.argv[1]); g = O.Grid.level("L1"); t = "beta0.01"
seq, ph = c[t + "/seq"], c[t + "/phase"]
E, W = g.E.reshape(g.ne, g.nw), g.W.reshape(g.ne, g.nw)
S = np.where(seq.reshape(E.shape) >= 0, seq.reshape(E.shape), np.nan).astype(float)
fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
im = ax.pcolormesh(E, W, S, cmap="Blues", shading="nearest", rasterized=True)
kept = seq.reshape(E.shape) < 0
ax.plot(E[kept], W[kept], ls="none", marker="s", ms=1.5, color="#eb6834", label="proper (retained)")
trap = ph.reshape(E.shape) == 1
ax.plot(E[trap], W[trap], ls="none", marker="s", ms=1.5, color="#0b0b0b", label="unreachable trap (last batch)")
for y in (-6.5, 6.5): ax.axhline(y, color="#52514e", lw=0.8, ls="--")
ax.text(-3.05, 6.7, "R: |ω| ≤ 6.5", color="#52514e")
ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-12, 12)
ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0 (upright)", "π/2", "π"])
ax.set_xlabel("e = θ − π (rad), periodic"); ax.set_ylabel("ω (rad/s)")
ax.set_title("L1 improper analysis, β = 1e-2: removal batch of each node\nsafety cascade from the |ω| = 12 edge (batch 0: no in-domain action) inward", loc="left", fontsize=9)
ax.legend(loc="lower left", fontsize=7, frameon=False)
fig.colorbar(im, ax=ax, label="removal batch (0 = first)")
fig.savefig(sys.argv[2], metadata={"Date": None, "Creator": None})
