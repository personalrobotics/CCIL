#!/bin/bash
# The freeze evidence of the Pendulum study (predeclaration section 11): every check, once, on one host, from one
# clean commit that combines the whole stack.
#
#     tools/m3_experiments/freeze_evidence.sh OUT cpu    # macOS; OUT is new and outside the repository
#     tools/m3_experiments/freeze_evidence.sh OUT cuda   # siddh; run the cpu host first or after, never mixed
#
# Each check writes OUT/<name>/report.json and OUT/<name>.log; OUT/status.txt lists every exit code, in the order the
# checks finish. Independent checks run concurrently (#137): up to JOBS (default 4) CPU checks at once and, on the
# CUDA host, one lane of GPU checks per device. The v6 driver runs and the timing parts then run alone, so the timing
# sees a quiet host. Afterwards each host audits the other's certificate bundle:
#
#     uv run --group evidence python tools/m3_certificates/audit_bundle.py --bundle OTHER/m3_certificates/bundle \
#         --out OUT/audit-of-OTHER
set -u
OUT=$1
DEV=$2
if [ -e "$OUT" ]; then echo "$OUT exists; evidence directories are never reused" >&2; exit 2; fi
mkdir -p "$(dirname "$OUT")"
case "$(cd "$(dirname "$OUT")" && pwd -P)/" in
    "$(git rev-parse --show-toplevel)"/*)  # checks scan the tree (dependency_gate reads every requirements file)
        echo "$OUT is inside the repository; write the evidence outside it" >&2; exit 2 ;;
esac
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
    uv run --group evidence ${uvopts[@]+"${uvopts[@]}"} python "tools/$tool/check.py" --out "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
    echo "$name EXIT=$?" >> "$OUT/status.txt"
}

legacy() {  # legacy NAME: a check of the legacy correct_il code, outside the project with its own pinned requirements
    # on the CPU: their pinned torch 2.1 has no kernels for current GPUs, and their CUDA cases expect none
    CUDA_VISIBLE_DEVICES= uv run --no-project --python 3.11 --with-requirements "tools/$1/requirements.txt" python "tools/$1/check.py" \
        --out "$OUT/$1" > "$OUT/$1.log" 2>&1
    echo "$1 EXIT=$?" >> "$OUT/status.txt"
}

slot() {  # wait for a free CPU slot: JOBS checks of this shell's own at once
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS:-4}" ]; do sleep 1; done
}

uv sync --group evidence > "$OUT/sync.log" 2>&1  # once, before the concurrent runs share the environment
# Foundations (M2), provenance (#127), conversion, labels, certificates, supervision, and the Pendulum environment:
# CPU checks, independent of each other and of the GPU lanes.
cpu_checks() {
    for t in m2_config m2_schemas m2_interfaces m2_pipelines m2_runs m3_provenance m2_packaging dependency_gate \
             legacy_conversion m3_labels m3_certificates m3_supervision; do
        slot; run "$t" -- "$t" &
    done
    for t in invariants legacy_serialization dynamics_checkpoint validation_split; do
        slot; legacy "$t" &
    done
    # The #24 section 3 acceptance checks; the two version 4 freeze-gate cases are recorded as skipped.
    slot; run m3_pendulum "${PEND[@]}" -- m3_pendulum --v6-freeze &
    slot; run vector_env "${VENV[@]}" -- vector_env &
    slot; run m3_newton_batched "${PEND[@]}" -- m3_newton_batched &
    if [ "$DEV" = cpu ]; then  # the device checks are CPU checks here
        slot; run m3_dynamics -- m3_dynamics --device cpu &
        slot; run m3_policy -- m3_policy --device cpu &
        slot; run m3_jacobian_chunks "${PEND[@]}" -- m3_jacobian_chunks --parts cpu --device cpu &
        slot; run m3_trainer_v2 -- m3_trainer_v2 --parts correctness,robustness,recipe,modes &
        slot; run m3_fit_v2 "${PEND[@]}" -- m3_fit_v2 --parts correctness,stage_hooks,clean_holdout,lipschitz &
        slot; run m3_experiments "${VENV[@]}" -- m3_experiments --runner --device cpu &
        slot; run m3_compile "${PEND[@]}" -- m3_compile --device cpu &
    fi
    wait
}
if [ "$DEV" = cuda ]; then
    # Two GPU lanes, one device each: the trainer's equivalence part is the longest check, so it has a device alone.
    ( export CUDA_VISIBLE_DEVICES=0
      run m3_trainer_v2 "${PEND[@]}" -- m3_trainer_v2 --device cuda --parts correctness,robustness,recipe,modes,equivalence
    ) &
    ( export CUDA_VISIBLE_DEVICES=1
      run m3_dynamics -- m3_dynamics --device cuda
      run m3_policy -- m3_policy --device cuda
      run m3_jacobian_chunks "${PEND[@]}" -- m3_jacobian_chunks --parts cpu,cuda --device cuda
      run m3_fit_v2 "${PEND[@]}" -- m3_fit_v2 --device cuda --parts correctness,stage_hooks,clean_holdout,lipschitz,pendulum,lipschitz_pendulum
      run m3_experiments "${VENV[@]}" -- m3_experiments --runner --device cuda
    ) &
fi
( cpu_checks ) &
wait
# The version 6 driver on the CUDA host, one shard per GPU. A --preflight run is the exact schedule with only the
# uncompiled budgets cut; the compilation check (#134) then requires every group of it compiled. A --smoke run
# (seeds 9101-9103, one epoch) gives the analysis a root to accept and, edited, to refuse (check.py --v6-smoke-root).
if [ "$DEV" = cuda ]; then
    RS=(uv run --group evidence "${VENV[@]}" python tools/m3_experiments/run_study.py)
    v6() {  # v6 ROOT MODE STEP: both shards of a phase step, or one step
        if [ "$3" = tuning ] || [ "$3" = study ]; then
            CUDA_VISIBLE_DEVICES=0 "${RS[@]}" "$3" --root "$1" --shard 0/2 "$2" > "$1-$3-0.log" 2>&1 &
            local p0=$!
            CUDA_VISIBLE_DEVICES=1 "${RS[@]}" "$3" --root "$1" --shard 1/2 "$2" > "$1-$3-1.log" 2>&1 &
            local p1=$!
            wait $p0; local e0=$?
            wait $p1; local e1=$?
            [ $e0 = 0 ] && [ $e1 = 0 ]
        else
            "${RS[@]}" "$3" --root "$1" "$2" > "$1-$3.log" 2>&1
        fi
    }
    v6_all() {  # v6_all ROOT MODE: every step in order, stopping at the first failure
        v6 "$1" "$2" tuning && v6 "$1" "$2" choose && v6 "$1" "$2" returns && v6 "$1" "$2" study && v6 "$1" "$2" verify
    }
    v6_all "$OUT/v6-preflight" --preflight
    echo "v6_preflight EXIT=$?" >> "$OUT/status.txt"
    run m3_compile "${PEND[@]}" -- m3_compile --device cuda --preflight-root "$OUT/v6-preflight"
    v6_all "$OUT/v6-smoke" --smoke
    echo "v6_smoke EXIT=$?" >> "$OUT/status.txt"
    run m3_experiments_analysis "${VENV[@]}" "${PEND[@]}" -- m3_experiments --v6-smoke-root "$OUT/v6-smoke"
fi
# Timing, last and one at a time, after every other check (CUDA host). The batched-fit gate is cited, not rerun: the confirmatory gate
# (b2cbf71) decided it, and the freeze records the fit-path diff from that commit (predeclaration section 11).
if [ "$DEV" = cuda ]; then
    run m3_trainer_v2_timing "${PEND[@]}" -- m3_trainer_v2 --device cuda --parts realworkload,scaling
    run m3_fit_v2_timing "${PEND[@]}" -- m3_fit_v2 --device cuda --parts timing
fi
echo ALL_DONE >> "$OUT/status.txt"
