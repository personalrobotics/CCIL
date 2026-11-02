"""Bit-identity of the blocked value iteration (version 6.1) against version 6 (644f5fb), on the real L1 grids.

    git show 644f5fb:tools/pendulum_oracle/oracle.py > OLD.py
    uv run --group evidence --with-requirements tools/vector_env/requirements.txt python docs/fixtures/evidence/pendulum-oracle/vi_equivalence.py OLD.py
"""
import importlib.util, sys, time
import numpy as np
sys.path.insert(0, "tools/pendulum_oracle")
import oracle as N
spec = importlib.util.spec_from_file_location("oracle_v6", sys.argv[1]); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
A, B = N.linearize()
for wmax in (N.OMEGA_MAX, N.OMEGA_MAX_CHECK):
    grid = N.Grid.level("L1", wmax=wmax)
    T = N.transitions(grid)
    for beta in (0.0, 1e-2, 1.0):
        L = N.lqr_bundle(beta, A, B)
        cont = N.boundary_values(grid, T, L, beta)
        for rows in (20_000, 7_777):
            N.VI_ROWS = rows
            for name, V0, mono in (("lower", np.zeros(len(grid.E)), True), ("upper", np.full(len(grid.E), 50.0), False)):
                t = time.time(); a = old.value_iteration(grid, T, L, beta, V0, cont, check_monotone=mono); ta = time.time() - t
                t = time.time(); b = N.value_iteration(grid, T, L, beta, V0, cont, check_monotone=mono, weights=grid.weights(T["e"], T["w"])); tb = time.time() - t
                same = all(np.array_equal(a[k], b[k], equal_nan=True) if isinstance(a[k], np.ndarray) else a[k] == b[k] for k in a)
                print(f"wmax {wmax} beta {beta} rows {rows} {name}: identical {same} iterations {a['iterations']} old {ta:.1f}s new {tb:.1f}s", flush=True)
                assert same
print("ALL IDENTICAL")
