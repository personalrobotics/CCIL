# Pendulum demonstration-count study: predeclaration, version 1 (#143)

**Status: version 1, binding when merged.** No policy has been trained on a subset of the corrected discontinuous demonstrations.

## Why

On the corrected demonstrations (`docs/fixtures/evidence/pendulum-corrected-demonstrator/study-evaluate.json`), CCIL − BC in upright occupancy is +0.208 [0.138, 0.277] on the continuous task (100 demonstrations) and +0.012 [−0.080, 0.104] on the discontinuous task (500 demonstrations). The two tasks differ in both the dynamics discontinuity and the number of demonstrations, so this result cannot separate the two.

**Hypothesis.** More demonstrations shrink CCIL's incremental benefit over BC, because BC's own coverage grows.

**Question.** On the discontinuous task, with the corrected demonstrator, is CCIL − BC larger with 100 demonstrations than with 500?

This study varies only the number of demonstrations on one task, so it tests the hypothesis directly. It does not isolate the discontinuity (see "Not concluded").

## What has been seen

- **Everything in the corrected-demonstrator study,** including both of its primary outcomes and every per-seed value. The hypothesis was formed after seeing them.
- **Historical data-size comparisons** (historical Part B, descriptive, `part-b.json`). These used the historical demonstrations on the discontinuous task, and at 95%:
  - CCIL − BC grew with data size instead of shrinking: +0.025 at 10%, +0.037 at 25%, +0.054 at 50% and +0.078 at 100%.
  - That is evidence against the hypothesis on the historical data. This study tests it on the corrected data.
- **Nothing on corrected subsets.** No subset of the corrected discontinuous dataset has been built, trained on or evaluated.

## Data

- **Source:** the canonical corrected discontinuous dataset, sha256 `a24d9adb0ee08911cadd76252534e114535e77b0ae264abb0a6c61c0c15af021` (500 episodes). It is verified by hash before use.
- **One subset per training seed s:** the first 100 episodes of `default_rng(SeedSequence([s, 9])).permutation(500)`.
  - This is v6's existing subset rule (`tools/m3_pendulum/study.py` `subset`) with fraction 0.2.
  - The subset is rebuilt on every call and must equal any existing file byte for byte.
- **BC and CCIL at seed s train on the same subset.** The 20 seeds give 20 different subsets.
- **Each subset is recorded by its sha256.** Every stage of a unit must trace its inputs to that subset.

## Training

- **Arms:**
  - `bc_100`: v6's BC arm settings;
  - `ccil_100`: v6's CCIL arm settings.
  - Both use the corrected study's shared settings (`tools/pendulum_eval/corrected_study.yaml`) unchanged. Only the dataset is replaced by the seed's subset.
- **Task:** discontinuous only.
- **Training seeds 1201–1220,** paired by seed with the corrected study's 500-demonstration BC and CCIL units.
- **No tuning** of anything.
- **Tooling.** Training uses the corrected study's tooling and the frozen ccil commit (6fbcce5). The tooling must take this plan, which needs a minimal change. That change is made in a separate reviewed PR before any training. It carries a smoke run with refusals, like the corrected study's (`study-smoke.json`), and is then frozen.
- **Order and provenance** are as in the corrected study: preflight, then train, audit and evaluate, each requiring its predecessor's committed report. The preflight and the audit also check that every unit's inputs trace to its seed's subset of the canonical dataset, and that no run ID or `.npz` numerical artifact is shared with v6 or with the corrected study.

## Evaluation

- **Starts:** the corrected study's fresh starts. Training seed s is evaluated on 100 episodes from `SeedSequence([s + 1000, 7])`, which gives evaluation seeds 2201–2220.
- **Evaluator and metric:** the batched evaluator, unchanged, and `pendulum_swingup_v2`. Upright occupancy is primary; height and final hold are secondary.
- **Noise-free only.** There is no noise evaluation.
- **The 500-demonstration values** are taken unchanged from `study-evaluate.json`. Nothing there is re-evaluated.
- **Audit of the saved rollouts:** an evaluation audit of the kind used for the corrected study reopens every saved rollout and reconstructs the report.

## Statistics

The interval is v6's union of BCa and paired t, with v6's resampling settings, over n = 20 training seeds, paired by seed. There are two primary analyses, so c = 1 − 0.05/2 = 0.975.

1. **(CCIL − BC)₁₀₀** in upright occupancy.
2. **(CCIL − BC)₁₀₀ − (CCIL − BC)₅₀₀,** paired by training seed. This is the test of the hypothesis: the hypothesis predicts a positive value.

**Outcomes** use v6's rules: "increases" when the lower endpoint is > 0, "decreases" when the upper endpoint is < 0, and otherwise "no detectable difference". A null outcome is reported with its interval, and it makes no claim that the difference is absent or unchanged.

**Failures and exclusions:** as v6 §9, applied to `bc_100` and `ccil_100`.
- An algorithmic failure (non-finite actions) makes the analysis inconclusive.
- More than two infrastructure exclusions in an arm makes it inconclusive.
- Fewer than 16 paired seeds makes it inconclusive.
- The 500-demonstration units are all `ok`.

**Secondary** (descriptive, 95%, no decisions):
- For each arm, the 100 − 500 change by seed, in occupancy, height and final hold.
- For CCIL's labels at 100 against 500 (the `label_accuracy` that each CCIL record already carries), by seed:
  - the acceptance rate;
  - the true one-step label error (mean and quantiles).
- (CCIL − BC)₁₀₀ on discontinuous beside (CCIL − BC) on continuous, both at 100 demonstrations. This comparison does not isolate the discontinuity: the two tasks also differ in their demonstration starts and their subsets.

## Not concluded

- **Whether the discontinuity itself reduces CCIL's benefit.** This study does not vary the dynamics.
- **Anything about other demonstration counts,** the modern recipe or robustness.
- **Anything general about CCIL.** The study covers one task, one demonstrator and one recipe.

## Order of work

1. Merge this predeclaration.
2. The tooling change and its smoke run, in a reviewed PR, then the freeze.
3. Preflight, train, audit and evaluate, each report reviewed before the next step.
4. The evaluation audit, then the analyses above.
