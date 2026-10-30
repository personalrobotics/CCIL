# Pendulum evaluation v2: historical results (#143)

The design is `docs/experiments/pendulum-eval-v2-predeclaration.md` (revision 2). These results use the historical demonstrations, the frozen v6 checkpoints and the task-aligned `pendulum_swingup_v2` metrics. The v6 analysis under the legacy return is unchanged and is reported elsewhere (`docs/experiments/pendulum-ccil-study-results.md`).

## The claim, and its limit

**With the historical demonstrations and the paper's training recipe, CCIL increases upright occupancy over BC on fresh evaluation starts.** On both tasks, the predeclared fresh-start validation (Part B) gives "CCIL improves on BC", and the positive control passes.

**This is not yet a general claim about CCIL.** Under the modern training recipe, the descriptive difference is small, and its interval includes zero on both tasks:

| Task | CCIL-modern − BC-modern, occupancy | 95% interval |
|---|---|---|
| Continuous | +0.011 | [−0.049, 0.070] |
| Discontinuous | −0.010 | [−0.052, 0.030] |

The paper-recipe advantage may therefore depend on the training recipe, and on the historical demonstrations, which come from a defective demonstrator (revision 2). The corrected-demonstrator study addresses the demonstrations, but nothing here addresses the recipe.

## Part B: predeclared fresh-start validation (evaluation seeds 2201–2220)

This is CCIL − BC in upright occupancy: the union of BCa and paired t at c = 0.975, over n = 20 training seeds.

| Task | Mean | Interval | Outcome | Positive control BC − BC 10% |
|---|---|---|---|---|
| Continuous | +0.158 | [0.107, 0.209] | CCIL improves on BC | +0.029 [0.003, 0.055], passes |
| Discontinuous | +0.077 | [0.020, 0.135] | CCIL improves on BC | +0.101 [0.051, 0.158], passes |

**Noise-free arms** (means over seeds):

| Arm | Continuous occupancy | Continuous final hold | Discontinuous occupancy | Discontinuous final hold |
|---|---|---|---|---|
| Zero action | 0.000 | 0.000 | 0.000 | 0.000 |
| BC | 0.174 | 0.000 | 0.252 | 0.101 |
| CCIL | 0.332 | 0.318 | 0.330 | 0.388 |
| CCIL-filtered | 0.169 | 0.084 | 0.263 | 0.113 |
| CCIL-noLip | 0.345 | 0.379 | 0.288 | 0.233 |
| BC-modern | 0.226 | 0.110 | 0.328 | 0.273 |
| CCIL-modern | 0.236 | 0.153 | 0.318 | 0.191 |

**Noise** (descriptive, with the channels separated). CCIL − BC is positive under action-only noise and negative under observation-only noise, on both tasks. The current observation mixture is not a physically justified sensor model (decision 3), so these results support no decision.

## Part A: post hoc, v6 rollouts, not confirmatory

All 1,440 stored v6 study rollouts were rescored after checking each one's sha256. Noise-free CCIL − BC in occupancy is +0.157 [0.111, 0.202] on continuous and +0.073 [0.019, 0.127] on discontinuous, consistent with Part B. CCIL-tuned and NoiseBC were selected on the legacy return, and are reported only here, descriptively.

## Evidence

- **`part-b-audit.json`** (`tools/pendulum_eval/audit_part_b.py`). Nothing is re-evaluated in this audit.
  - **The run commit:** 70e25a4, which contains the Part B script.
  - **The run log's erroneous commit line:** 57faa1d. The launcher expanded it in the main checkout when the script was written, and that commit does not contain the Part B script. The original log is preserved unchanged, and its sha256 is recorded.
  - **Hashes:** all 480 reused checkpoints match their v6 records, and every saved Part B rollout matches the report.
  - **Recomputation:** every section of `part-b.json` (per-seed scores, decisions, comparisons, summaries and failures) is recomputed from the saved rollouts, and each is equal.
- **Before any fresh-start rollout,** the evaluation loop reproduced 16 stored v6 rollouts bit for bit. Part A, regenerated, is byte-identical.

## Commanded actions beyond ±3

- **Fraction:** CCIL-trained policies command |a| > 3 on 15–20% of noise-free steps. The exception is CCIL-filtered: 0% on continuous and 0.9% on discontinuous. BC policies never do.
- **Size:** the excess is tiny and fixed. Every out-of-range step sits at the same ceiling: at most 5.1e-5 on continuous and 5.4e-4 on discontinuous (`part-b-audit.json`).
- **Source:** each CCIL-trained checkpoint's action scaler has bounds just beyond ±3 (up to 3.00005 continuous, 3.00054 discontinuous), matching the excess. BC's bounds are exactly ±3. The excess is therefore the policy's output range, inherited from its training labels. It is not float rounding.

**Clipped-action sensitivity** (descriptive; `part-b-clip-sensitivity.json`, `tools/pendulum_eval/clip_sensitivity.py`). Every reused unit was re-evaluated noise-free on the same fresh starts, with each commanded action clipped to ±3:
- The primary outcome is unchanged on both tasks: +0.158 [0.107, 0.209] and +0.078 [0.020, 0.136]. The positive control still passes.
- The largest per-seed change in any arm's occupancy is 0.027 (discontinuous CCIL-modern). BC arms do not change.
- CCIL-modern − BC-modern stays near zero: +0.011 [−0.049, 0.070] and −0.012 [−0.053, 0.027].

## Not concluded here

- Anything about the expert. No fresh-start expert reference exists (revision 2).
- Anything about robustness. The noise model is not physically scaled.
- Whether the paper-recipe effect survives corrected demonstrations. That is the separate corrected-demonstrator study.
