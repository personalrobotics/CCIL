# Pendulum evaluation v2: predeclaration of learned-policy scoring (#143)

**Status: draft for review.** No learned-policy score has been computed under `pendulum_swingup_v2`. None may be computed until this document is merged. Whatever this document leaves open stays unscored until it is decided here, before any score is read.

## What has been seen

- **The v6 analysis under the legacy return:** `docs/fixtures/evidence/pendulum-study-v6/analysis.json`, with its per-seed returns.
- **The v6 behavioral diagnostics** (`docs/experiments/pendulum-v6-diagnostics.md`): per-episode categories at a 0.3 rad threshold, not the 8° threshold used here, for zero, BC, NoiseBC, CCIL and CCIL-tuned.
  - These are related to v2's upright time, but they are not v2 scores. They are disclosed here because they inform expectations. For example, noise-free CCIL ended upright in 34% (continuous) and 42% (discontinuous) of episodes, against BC's 0.1% and 13%.
  - They also showed that CCIL's loss under evaluation noise comes from observation noise: with action noise alone, CCIL − BC was positive.
- **The expert-alignment gate:** `docs/fixtures/evidence/pendulum-eval-v2/expert-alignment.json`. It covers the demonstrations and same-start zero action only.

## Profile

`pendulum_swingup_v2`, `pendulum_eval.profiles/1` (`tools/pendulum_eval/profiles.py`):
- **Upright:** within 8° of vertical.
- **Hold:** the final 100 steps (2 s).
- **States scored:** the T pre-action states of each episode.
- **Actions:** applied unclipped, as in v6. Commanded and applied out-of-range fractions are reported.
- **Not used:** the legacy CCIL return and the Gymnasium Pendulum return are reported only as named historical metrics.

## Part A: post hoc rescoring of v6 (descriptive)

- **What is scored:** every stored v6 study rollout, every arm and every noise level, at study seeds 1201–1220 with their stored initial states.
- **No rerun.** Scores are computed from the stored states and actions. Each rollout's sha256 is checked against its record first.
- **Labeled post hoc.** Every table carries "post hoc, v6 rollouts, not confirmatory". The v6 analysis and its decision are unchanged and are reported beside these scores, never replaced by them.
- **Statistics:** the same paired union interval as v6, at 95%, descriptive. There is no decision and no multiplicity claim.

## Part B: confirmatory comparison on fresh evaluation starts

**Question.** On each task, does CCIL change upright occupancy relative to BC, noise-free, under the paper's training settings?

**Checkpoints.** The frozen v6 checkpoints of training seeds 1201–1220. They are reused only where neither training nor model selection used the legacy return:

| Arm | Reused | Why |
|---|---|---|
| BC, CCIL | yes | No return-based choice |
| CCIL-filtered, CCIL-noLip, BC and CCIL at 10%, 25% and 50% | yes | No return-based choice |
| BC-modern, CCIL-modern | yes | Learning rate chosen on validation loss only |
| NoiseBC, CCIL-tuned | **excluded** | Noise scale and action noise were chosen on the legacy return. Retuning them under v2 would need training, and stays open (decision 2). |

**Evaluation starts.** Training seed s is evaluated on 100 fresh episodes. Their initial states come from `SeedSequence([s + 1000, 7])`, which gives evaluation seeds 2201–2220. These starts have never been used. They use the same distribution as v6: θ ~ U(−1, 1), θ̇ ~ U(−0.5, 0.5).

**Evaluation.** The batched evaluator (`tools/vector_env/pendulum.py`) with v6's dynamics, horizon of 500 steps, action semantics and discontinuity.

**Primary decision.**
- Per task: the paired difference CCIL − BC in upright occupancy, n = 20 training seeds.
- The interval is the union of BCa and paired t, at c = 0.975 (Bonferroni over the two tasks), with v6's resampling settings.
- The outcomes are "improves", "worse" and "no detectable difference", with v6's rules.

**Positive control.** BC − BC 10% in upright occupancy, with the same rule as v6. It governs only how a null result is read.

**Secondary (descriptive, 95%):**
- normalized height and final-hold rate for the primary pair;
- every reused arm against BC;
- CCIL-modern − BC-modern;
- the data-size pairs;
- the behavior diagnostics.

**Zero action.** Zero action is scored on the same fresh starts, as a reference.

**Failures and exclusions.** As v6 §9: non-finite actions count as algorithmic failures. There are no other exclusions.

**Noise.** It is not part of the decision. Robustness is reported descriptively, with observation noise and action noise as separate channels at v6's levels, from the same checkpoints and fresh starts. Whether the noise model itself (`o = (1 − n)·s + n·U(bounds)`) is a sensible perturbation is decision 3.

## Open decisions (to settle in review, before any score)

1. **The expert on identical starts.**
   - The demonstrations come from the paper's energy-shaping plus LQR controller, but the controller is not in the repository. Only its trajectories are. So the expert cannot be scored on the fresh starts, which #143's acceptance criteria require.
   - Options:
     - (a) Reconstruct the controller, validate it by replaying each demonstration from its first state, then score it on the fresh starts.
     - (b) Report the demonstrations on their own starts, labeled as different starts.
   - Recommended: (a). Until it is decided, the expert is reported as (b), labeled.
2. **NoiseBC and CCIL-tuned.** Retune under v2 using validation-free selection on fresh tuning seeds (this needs training), or leave them excluded. Recommended: leave them excluded from Part B, and report Part A descriptively.
3. **The noise model.** Keep v6's levels and model as a descriptive robustness report, or declare a physically scaled perturbation. Recommended: keep it descriptive in this round, with the two channels separated, and define a new noise model separately, before it is ever used for a decision.

## Artifacts

- **Scripts:** a committed scorer under `tools/pendulum_eval/`, run from a clean commit. Its structured reports are committed under `docs/fixtures/evidence/pendulum-eval-v2/`, and each report reconciles every episode and seed against its source records.
- **Ordering:** Part A and Part B are run only after this document is merged.
