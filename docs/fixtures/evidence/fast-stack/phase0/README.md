# Phase-0 baseline: the current stack on Pendulum (#114)

The contract is fixed in #114 before any run:
- seed 501;
- per task, one representative seed group, run in-process exactly as the #24 drivers configure it;
- cold repetitions, each a fresh process with a fresh stage store;
- GPU-synchronized stage boundaries;
- **3 repetitions, the median reported**, with the minimum and maximum.

Raw artifacts are unmodified:

| Path | Contents |
|---|---|
| `20261026T233124Z-baseline/<task>-rep<k>/timing.json` | Stage events, per-epoch training times, workloads, resolved configurations, input hashes |
| `20261026T233124Z-baseline/<task>-rep<k>/gpu.csv` | 250 ms GPU utilization and memory trace |
| `20261026T233124Z-baseline/<task>-rep<k>/rss.csv` | 250 ms process RSS trace |
| `20261026T233124Z-baseline/meta.json` | Commit, host, per-repetition load average |
| `20261026T233124Z-baseline/report.json` | Medians and projections |
| `20261026T233124Z-spike/`, `20261026T234610Z-spike-gpu1/` | The #108 engine spike |
| `*.log` | Driver logs |

**Reproduce** (siddh, from commit `6ec1544`):

```bash
uv run --group evidence --with-requirements tools/m3_pendulum/requirements.txt \
    python tools/fast_stack/baseline.py --out <new-dir> --reps 3 --gpu 0
uv run --group evidence python tools/fast_stack/spike_trainer.py --out <new-dir> --gpu 0
```

## Setup

- **Host:** siddh, 2× NVIDIA RTX PRO 6000 Blackwell Max-Q (driver 595.99.02), 64 cores.
- **Software:** torch 2.14.1+cu130, Python 3.11.17.
- **Code:** commit `6ec1544`, with `ccil` `src` identical to 847b060 (trainer v1).
- **Device:** GPU 0, one process.

## Measured: cold seed group per task

| Task | Median | Min–max | Interpreter, import and CUDA init |
|---|---|---|---|
| Continuous | **308.4 s** | 299.5–311.9 | 1.0 s |
| Discontinuous | **427.7 s** | 424.8–436.6 | 1.0 s |

## Measured: stages (median of 3)

Peak memory is PyTorch's peak allocated memory, reset at each stage.

| Stage | Continuous | Discontinuous | Peak GPU memory (cont / disc) | Mean GPU utilization |
|---|---|---|---|---|
| `transition.fit` | 25.1 s | 115.1 s | 0.4 / 0.3 GB | 39–47% |
| `labels.generate` (Newton) | 1.2 s | 6.2 s | 0.5 / 0.3 GB | 3–10% |
| `labels.certify` (paper proxy) | 4.1 s | 20.0 s | **21.6 / 54.2 GB** | 2–9% |
| `policy.train`, BC | 8.1 s | 40.9 s | <0.1 GB | 30% |
| `policy.train`, NoiseBC | 14.7 s | 75.0 s | <0.1 GB | 17–18% |
| `policy.train`, CCIL | **169.5 s** | 81.6 s (balanced) | 0.1 / 0.2 GB | 31–33% |
| `policy.train`, CCIL-filtered | 56.6 s | 59.1 s | <0.1 GB | 29–32% |
| One evaluation rollout (100 episodes × 500 steps) | 3.2 s | 3.3 s | – | 1–3% |

**Steady-state policy epoch** (median over epochs after the first):

| Arm | Continuous | Discontinuous |
|---|---|---|
| BC | 0.08 s | 0.41 s |
| NoiseBC | 0.15 s | 0.74 s |
| CCIL | 1.69 s | 15.7 s |
| CCIL-filtered | 0.56 s | 10.8 s |

**Reading:**
- **Policy training dominates.** It runs at about 30% GPU utilization: the host-bound per-step loop of #108.
- The dynamics fit is second, mainly on discontinuous.
- Newton label generation is cheap here, at 1–6 s. The earlier per-row-loop concern (#110) matters at larger scale, not on Pendulum.
- Certification is fast but needs 54 GB (#109, #110).
- Evaluation is cheap per rollout, but #24 needs about 1,100 of them.

## PROJECTED, not measured: the full #24 design

Serial, one process, one GPU, computed from the medians above with the formula in `report.json`:

| | Tuning, 3 seeds | Study, 20 seeds |
|---|---|---|
| Continuous | 54 min | 304 min |
| Discontinuous | 53 min | 331 min |

**Total: about 12.4 h serial.** The v4 run itself (10 processes on one GPU) projected 15–20 h. That is observed contention data, reported separately in #24.

## The #108 spike (measured)

Group optimizer-step time on one shared table of 1.05 M rows, GPU 0 (`20261026T233124Z-spike`):

| Engine | M = 1 | M = 20 | M = 64 |
|---|---|---|---|
| v1 (`torch.optim.Adam`, no syncs) | 1.88 ms | – | – |
| stacked, eager | 0.72 | 0.71 | 0.76 |
| stacked, compiled | 0.34 | 0.39 | 0.55 |
| vmap, eager | 1.29 | 1.18 | 1.23 |
| **vmap, compiled** | **0.15** | **0.19** | 0.60 |

- **Decision:** vmap + `torch.compile`. Peak memory is at most 0.37 GB at M = 64.
- After 200 steps, model 0's parameters differ between M = 1 and M = 20 by 4e-5 (eager) and 4e-3 (compiled). They differ from v1 by about 5% of the largest parameter, which is Adam's sign-like update amplifying float32 rounding. #108 therefore relies on engine-correctness checks and statistical equivalence, not parameter agreement.
- The GPU-1 run (`20261026T234610Z-spike-gpu1`) was made concurrently with the baseline. The chosen engine, vmap compiled, agrees within 3% at every M; other engines differ by up to 15%. The engine ranking is the same in both runs.

## Conditions and incidents (disclosed)

- **Another project's job** (`ssrobot-207-choose`) ran on siddh from 16:33. It used about 9% of one core and no GPU. The load average stayed between 0.9 and 1.5 on 64 cores (recorded before each repetition in `meta.json`).
- **Concurrent work on GPU 1:** the spike at 16:46–16:50, during continuous repetition 2, and a trainer-v2 test at 17:04–17:08, during discontinuous repetition 3. GPU 0 was used only by the baseline.
- **Checkout incident.** At 17:04:08 the baseline's working tree was switched to the trainer-v2 branch, and restored to `6ec1544` 11 s later. Discontinuous repetition 3 was running; it had started at 17:03:21 and loaded its code at start.
  - In that window the trees differed only by a new file the baseline never imports.
  - The repetition's time (428 s) lies between repetitions 1 and 2 (437, 425 s).
- **Repetition spread** is under 4% on both tasks, so none of these conditions shows up in the measurements.
- **Device-memory caveat.** `nvidia-smi` memory (`device_mem_max_mb`) is the process's reserved memory, which PyTorch's caching allocator keeps after a stage. After certification it stays at the certification peak. Per-stage memory is the PyTorch peak allocated, in the table above.
