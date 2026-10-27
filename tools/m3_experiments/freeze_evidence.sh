#!/bin/bash
# The freeze evidence of the Pendulum study (predeclaration section 11): every check, once, on one host, from one
# clean commit that combines the whole stack.
#
#     tools/m3_experiments/freeze_evidence.sh OUT cpu    # macOS
#     tools/m3_experiments/freeze_evidence.sh OUT cuda   # siddh; run the cpu host first or after, never mixed
#
# Each check writes OUT/<name>/report.json and OUT/<name>.log; OUT/status.txt lists every exit code. The CUDA-only
# timing and gate parts run last, one at a time, so they see a quiet device. Afterwards each host audits the other's
# certificate bundle:
#
#     uv run --group evidence python tools/m3_certificates/audit_bundle.py --bundle OTHER/m3_certificates/bundle \
#         --out OUT/audit-of-OTHER
set -u
OUT=$1
DEV=$2
if [ -e "$OUT" ]; then echo "$OUT exists; evidence directories are never reused" >&2; exit 2; fi
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then echo "the tree is not clean" >&2; exit 2; fi
mkdir -p "$OUT"
{ echo "commit $(git rev-parse HEAD)"; echo "device $DEV"; uptime; } > "$OUT/status.txt"
PEND=(--with-requirements tools/m3_pendulum/requirements.txt)
VENV=(--with-requirements tools/vector_env/requirements.txt)

run() {  # run NAME [uv options] -- TOOL [tool options]
    local name=$1; shift
    local uvopts=()
    while [ "$1" != "--" ]; do uvopts+=("$1"); shift; done
    shift
    local tool=$1; shift
    uv run --group evidence "${uvopts[@]}" python "tools/$tool/check.py" --out "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
    echo "$name EXIT=$?" >> "$OUT/status.txt"
}

# Foundations (M2), provenance (#127) and conversion.
for t in m2_config m2_schemas m2_interfaces m2_pipelines m2_runs m3_provenance m2_packaging dependency_gate invariants \
         legacy_serialization legacy_conversion dynamics_checkpoint validation_split; do
    run "$t" -- "$t"
done
# Labels, certificates, supervision, dynamics, policy.
run m3_labels -- m3_labels
run m3_certificates -- m3_certificates
run m3_supervision -- m3_supervision
run m3_dynamics -- m3_dynamics --device "$DEV"
run m3_policy -- m3_policy --device "$DEV"
# The Pendulum environment and the fast stack.
run m3_pendulum "${PEND[@]}" -- m3_pendulum
run vector_env "${VENV[@]}" -- vector_env
run m3_newton_batched "${PEND[@]}" -- m3_newton_batched
if [ "$DEV" = cuda ]; then
    run m3_jacobian_chunks "${PEND[@]}" -- m3_jacobian_chunks --parts cpu,cuda --device cuda
    run m3_trainer_v2 "${PEND[@]}" -- m3_trainer_v2 --device cuda --parts correctness,robustness,recipe,modes,equivalence
    run m3_fit_v2 "${PEND[@]}" -- m3_fit_v2 --device cuda --parts correctness,stage_hooks,clean_holdout,lipschitz,pendulum,lipschitz_pendulum
else
    run m3_jacobian_chunks "${PEND[@]}" -- m3_jacobian_chunks --parts cpu --device cpu
    run m3_trainer_v2 -- m3_trainer_v2 --parts correctness,robustness,recipe,modes
    run m3_fit_v2 "${PEND[@]}" -- m3_fit_v2 --parts correctness,stage_hooks,clean_holdout,lipschitz
fi
run m3_experiments "${VENV[@]}" -- m3_experiments --runner --device "$DEV"
# Timing, last and one at a time (CUDA host). The batched-fit gate is cited, not rerun: the confirmatory gate
# (b2cbf71) decided it, and the freeze records the fit-path diff from that commit (predeclaration section 11).
if [ "$DEV" = cuda ]; then
    run m3_trainer_v2_timing "${PEND[@]}" -- m3_trainer_v2 --device cuda --parts realworkload,scaling
    run m3_fit_v2_timing "${PEND[@]}" -- m3_fit_v2 --device cuda --parts timing
fi
echo ALL_DONE >> "$OUT/status.txt"
