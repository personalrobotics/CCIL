# Pendulum evaluation v2: predeclaration of learned-policy scoring (#143)

**Status: revision 2, binding when merged.** Revision 1 was merged in PR #144. Revision 2 changes only the expert reference (see "Revision 2" below). No learned-policy score has been computed under `pendulum_swingup_v2`, and no Part B evaluation has been run. Neither may run until both this revision and the separate repair-study predeclaration (`docs/experiments/pendulum-corrected-demonstrator-predeclaration.md`) are merged, so that the repair study's design cannot react to the historical fresh-start results.

## Revision 2: the historical expert reconstruction is terminated

1. **Revision 1 failed as declared.** The frozen reconstruction (`docs/experiments/pendulum-expert-reconstruction.md`, committed at bb18bf0 before any comparison) failed its per-state check for all four declared variants on both tasks (`docs/fixtures/evidence/pendulum-eval-v2/expert-reconstruction.json`). The closed-loop replay was not run, as the rules require.
2. **The Appendix D controller, as printed, does not reproduce the demonstrations.**
3. **A wrapping inconsistency accounts for all in-window disagreements.** The generator tested the LQR window on the angle wrapped to [0, 2π), but used the raw atan2 angle in (−π, π] in the LQR term. On the θ > π side of the window, the term is about −20.11·(−2π), so the stored action is +3 there. This is post hoc and exploratory (`tools/pendulum_eval/expert_diagnosis.py`, `docs/fixtures/evidence/pendulum-eval-v2/expert-diagnosis.json`).
   - The large disagreements are 384 states (continuous) and 2,256 (discontinuous). All of them, and only they, are on that side of the window.
   - The inconsistent rule leaves no large disagreement.
   - Float32 evaluation plausibly explains the remaining small residuals: each is within about 1.1 times a coarse float32 rounding scale of its formula.
   - Demonstrations that visit the affected side have lower occupancy and every final-hold failure. This is a descriptive association. The defect may contribute to the expert's occupancy shortfall, but that causal claim is not established.
4. **The historical expert reconstruction is terminated.** The defective controller is not adopted as the expert, check 1 is not relaxed, and no defect-faithful replay is run.
5. **No fresh-start historical expert score will be produced.**
6. **Part A and Part B continue without an expert reference.**
   - Part A's rescoring remains descriptive.
   - Part B remains CCIL versus BC on fresh starts.
   - Zero action remains the only fresh-start reference.
7. **Demonstration scores** (the expert-alignment gate, and any later report) are labeled **"historical demonstrator, original starts"**.

A corrected demonstrator, with the wrapping made consistent, is a separate study with its own predeclaration (`docs/experiments/pendulum-corrected-demonstrator-predeclaration.md`). It is not an expert reference for Part B.

## What has been seen

- **The v6 analysis under the legacy return:** `docs/fixtures/evidence/pendulum-study-v6/analysis.json`, with its per-seed returns.
- **The v6 behavioral diagnostics** (`docs/experiments/pendulum-v6-diagnostics.md`): per-episode categories at a 0.3 rad threshold, not the 8° threshold used here, for zero, BC, NoiseBC, CCIL and CCIL-tuned.
  - These are related to v2's upright time, but they are not v2 scores. They are disclosed here because they inform expectations. For example, noise-free CCIL ended upright in 34% (continuous) and 42% (discontinuous) of episodes, against BC's 0.1% and 13%.
  - They also showed that CCIL's loss under evaluation noise comes from observation noise: with action noise alone, CCIL − BC was positive.
- **The expert-alignment gate:** `docs/fixtures/evidence/pendulum-eval-v2/expert-alignment.json`. It covers the demonstrations and same-start zero action only.
- **The failed expert reconstruction and its post hoc diagnosis** (revision 2). These are demonstrator analyses only; no learned policy was evaluated.

## Profile

`pendulum_swingup_v2`, `pendulum_eval.profiles/1` (`tools/pendulum_eval/profiles.py`):
- **Upright:** within 8° of vertical.
- **Hold:** the final 100 steps (2 s).
- **States scored:** the T pre-action states of each episode.
- **Actions:** applied unclipped, as in v6. Commanded and applied out-of-range fractions are reported.
- **Not used:** two named historical metrics, reported but never decided on:
  - the legacy CCIL return;
  - Gymnasium Pendulum-v1's cost formula on the CCIL trajectory. This is not a Gymnasium return: the trajectory follows CCIL's dynamics and may apply torques outside Gymnasium's ±2 limit, which the formula clips only inside the cost.
- **Ingress:** states must be (E, T, 3), applied actions exactly (E, T, 1), and commanded actions, when recorded, the same shape. Episodes shorter than the 100-step hold window, and any non-finite state or action, are refused. A non-finite rollout is recorded as an algorithmic failure (see Failures and exclusions), never scored.

## Part A: post hoc rescoring of v6 (descriptive)

- **What is scored:** every stored v6 study rollout, every arm and every noise level, at study seeds 1201–1220 with their stored initial states.
- **No rerun.** Scores are computed from the stored states and actions. Each rollout's sha256 is checked against its record first.
- **Labeled post hoc.** Every table carries "post hoc, v6 rollouts, not confirmatory". The v6 analysis and its decision are unchanged and are reported beside these scores, never replaced by them.
- **Statistics:** the same paired union interval as v6, at 95%, descriptive. There is no decision and no multiplicity claim.

## Part B: predeclared fresh-start validation of CCIL versus BC

**What this is.** Part B is prospective with respect to the evaluation starts (seeds 2201–2220), which have never been used. It is not an untouched original confirmatory test. The 0.3 rad behavioral diagnostics and the ended-upright differences listed above (close to final hold) were already seen, and they motivated this metric correction. Part B is a predeclared validation, or replication, of those observations on fresh starts under the corrected metric.

**Question.** On each task, does CCIL change upright occupancy relative to BC, noise-free, under the paper's training settings?

**Checkpoints.** The frozen v6 checkpoints of training seeds 1201–1220. They are reused only where neither training nor model selection used the legacy return:

| Arm | Reused | Why |
|---|---|---|
| BC, CCIL | yes | No return-based choice |
| CCIL-filtered, CCIL-noLip, BC and CCIL at 10%, 25% and 50% | yes | No return-based choice |
| BC-modern, CCIL-modern | yes | Learning rate chosen on validation loss only |
| NoiseBC, CCIL-tuned | **excluded** | Noise scale and action noise were chosen on the legacy return (decision 2) |

**Evaluation starts.** Training seed s is evaluated on 100 fresh episodes. Their initial states come from `SeedSequence([s + 1000, 7])`, which gives evaluation seeds 2201–2220. These starts have never been used. They use the same distribution as v6: θ ~ U(−1, 1), θ̇ ~ U(−0.5, 0.5).

**Evaluation.** The batched evaluator (`tools/vector_env/pendulum.py`) with v6's dynamics, horizon of 500 steps, action semantics and discontinuity.

**Primary analysis.**
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

**References.**
- Zero action is scored on the same fresh starts.
- No expert reference is scored on the fresh starts (revision 2). The demonstrations are reported only as "historical demonstrator, original starts".

**Failures and exclusions.** As v6 §9: non-finite actions count as algorithmic failures. There are no other exclusions.

**Noise.** It is not part of the decision. Robustness is reported descriptively, with observation-only and action-only perturbations as separate channels at v6's levels, from the same checkpoints and fresh starts (decision 3).

## Decisions (settled in review of PR #144, before any learned-policy score)

1. **Reconstruct the expert.** *Superseded by revision 2: the reconstruction ran as declared, failed, and is terminated.*
   - Use the paper's exact hybrid controller: energy shaping with an LQR balance controller, including its angle wrapping, its `|θ − π| < 0.1` switch, and the action-bound behavior that the stored data establishes.
   - Validate it in two independent ways:
     - **Per-state actions.** Evaluate the controller at every stored demonstration state, and compare its action directly with the stored action.
     - **Closed-loop replay.** Replay from every demonstration's first state, and compare the complete state and action trace, including the discontinuous task's velocity flips.
   - **Tolerances are frozen in a commit before either comparison is run.**
   - Only after both checks pass is the expert evaluated on the fresh starts. Until then, its demonstration-only scores stay labeled "different starts".
2. **NoiseBC and CCIL-tuned are excluded from Part B.**
   - Their settings were selected on the defective legacy return.
   - Their frozen v6 rollouts are rescored in Part A as descriptive evidence only.
   - No compute is spent retuning them for this comparison. A later study may retune every compared method under a valid, common selection rule.
3. **Robustness stays descriptive, with the channels separated.**
   - Observation-only and action-only perturbations are reported independently, and neither enters a decision.
   - The current observation mixture, `o = (1 − n)·s + n·U(bounds)`, is not a physically justified sensor model.
   - A physically scaled noise model is defined and predeclared in separate work before robustness is used for any scientific decision.

## Artifacts

- **Scripts:** a committed scorer under `tools/pendulum_eval/`, run from a clean commit. Its structured reports are committed under `docs/fixtures/evidence/pendulum-eval-v2/`, and each report reconciles every episode and seed against its source records.
- **Ordering:** Part A and Part B are run only after this revision and the repair-study predeclaration are both merged.
