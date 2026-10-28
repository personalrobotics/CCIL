#!/usr/bin/env bash
# v6 Pendulum study (predeclaration v6, frozen at 6fbcce5; main d4ef5d1). Real runs only.
set -u
TS="$1"
RUNS=$HOME/code/CCIL/runs
R=$RUNS/$TS-v6-study
LOG=$RUNS/$TS-v6-study.log
cd $HOME/code/CCIL-v6 || exit 1
export PATH=$HOME/.local/bin:$PATH
{
  echo "commit $(git rev-parse HEAD)"
  echo "dirty $(git status --porcelain --untracked-files=all | wc -l)"
  echo "load $(cat /proc/loadavg)"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
  uv run --group evidence --with-requirements tools/vector_env/requirements.txt python -c 'import torch,numpy,scipy;print("torch",torch.__version__,"numpy",numpy.__version__,"scipy",scipy.__version__)'
} > "$LOG" 2>&1
mkdir "$R" || { echo "root exists EXIT=1" >> "$LOG"; exit 1; }
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 1 > "$RUNS/$TS-v6-gpu-util.csv" &
SMI=$!
RS=(uv run --group evidence --with-requirements tools/vector_env/requirements.txt python tools/m3_experiments/run_study.py)
step() {
  echo "$(date -u +%FT%TZ) start $1" >> "$LOG"
  if [ "$1" = tuning ] || [ "$1" = study ]; then
    CUDA_VISIBLE_DEVICES=0 "${RS[@]}" "$1" --root "$R" --shard 0/2 > "$R-$1-0.log" 2>&1 & p0=$!
    CUDA_VISIBLE_DEVICES=1 "${RS[@]}" "$1" --root "$R" --shard 1/2 > "$R-$1-1.log" 2>&1 & p1=$!
    wait $p0; e0=$?; wait $p1; e1=$?
    rc=$(( e0 || e1 ))
  else
    "${RS[@]}" "$1" --root "$R" > "$R-$1.log" 2>&1; rc=$?
  fi
  echo "$(date -u +%FT%TZ) end $1 EXIT=$rc" >> "$LOG"
  return $rc
}
step tuning && step choose && step returns && step study && step verify
rc=$?
kill $SMI
if [ $rc = 0 ]; then
  echo "$(date -u +%FT%TZ) start analyze" >> "$LOG"
  uv run --group evidence --with-requirements tools/vector_env/requirements.txt --with-requirements tools/m3_pendulum/requirements.txt \
    python tools/m3_experiments/analyze_v6.py --root "$R" --out "$RUNS/$TS-v6-analysis.json" > "$R-analyze.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) end analyze EXIT=$rc" >> "$LOG"
fi
echo "EXIT=$rc" >> "$LOG"
