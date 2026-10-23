# Dynamics checkpoints

This describes how learned transition models are saved and loaded (#12). The
implementation is in [`correct_il/models/nn_dynamics.py`](../../correct_il/models/nn_dynamics.py)
(`checkpoint_name`, `save_world_model`, `load_world_model`, `CheckpointError`).

## Format

A checkpoint is a `torch.save` payload that loads with
`torch.load(..., weights_only=True)`. Loading it cannot execute code. It holds:

| Key | Contents |
|---|---|
| `meta` | `format: "ccil-dynamics/1"`, `state_dim`, `act_dim`, `hidden_size`, `activation`, the full `d_config`, `training` (the module mode) and `dtype: "float32"`. |
| `dynamics_net` | The `state_dict` and the normalization tensors (`s_shift`, `s_scale`, `a_shift`, `a_scale`, `out_shift`, `out_scale`). The normalization tensors are plain attributes, so `state_dict()` omits them. |
| `learnable_lambda` | The same, present only when `lipschitz_type` contains `slack`. |

Tensors are stored on CPU as float32 and loaded as float32.

## Names

`checkpoint_name(backward)` returns `dynamics.pt` or `dynamics_backward.pt`.
Training (`dynamics.backward`) and label generation (`aug.type: backward_model`)
both use it. A backward model is fitted on reversed transitions,
(s_{t+1}, a_t) → s_t.

Legacy `dynamics.pkl` pickles are not loaded. Retrain the model, or convert it
through the trusted legacy path in #15.

## Loading and devices

`load_world_model(path, device)` reads to CPU, validates the payload, rebuilds
the model from `meta`, loads the state dicts strictly, restores the module mode
and moves everything to `device`. Label generation uses
`"cpu" if config.no_gpu else "cuda"`, the same rule as training.

The module mode is restored because spectral-norm layers run a power iteration
on every forward pass in training mode. Restoring the mode keeps predictions
identical across a save and load. Whether inference *should* run in training
mode is a separate correctness question, outside #12.

## Errors

Every failure raises `CheckpointError` with an explicit message. Nothing falls
back silently, and a failed save never modifies an existing file.

| Case | Message contains |
|---|---|
| The file is unreadable, not a checkpoint, or a legacy pickle | `cannot read` (the pickle is never executed) |
| The format is unknown or missing | `unsupported checkpoint format` |
| The shapes don't match `meta` | `incompatible architecture` |
| The stored networks don't match `lipschitz_type` | `do not match` |
| `cuda` is requested where CUDA is unavailable | `CUDA is unavailable` |
| The save fails partway | The previous file is unchanged, and the temporary file is removed. |

## Supported device cases and tolerances

| Case | Tolerance | Observed max abs difference | Verified on |
|---|---|---|---|
| Saved and loaded on CPU | 1e-6 | 0 | macOS arm64, torch 2.1.0, no CUDA; Linux, torch 2.7.1+cu128 |
| Saved on CUDA, loaded on CUDA | 1e-6 | 0 | Linux, 2 × RTX PRO 6000 Blackwell, torch 2.7.1+cu128 |
| Saved on CUDA, loaded on CPU | 1e-5 | 6.0e-8 | the same |
| Saved on CPU, loaded on CUDA | 1e-5 | 6.0e-8 | the same |

Results across accelerators are not bitwise identical, and nothing promises
they are.

## Reproduce

```bash
uv run --no-project --python 3.11 --with-requirements tools/dynamics_checkpoint/requirements.txt \
  python tools/dynamics_checkpoint/check.py --out <new-dir>
```

On a CUDA host, the same script also runs the cross-device cases. The torch
version must support the GPU; Blackwell needs CUDA 12.8 or newer.
`report.json` records every case, the measured differences, the tolerances,
the code commit and the package versions.

The evidence is in [`evidence/dynamics-checkpoints/`](evidence/dynamics-checkpoints/):
- `cpu-macos/report.json`: 13/13 cases pass.
- `gpu-linux/report.json`: 24/24 cases pass, cross-device included.

Both runs cover all four `lipschitz_type` variants (`none`, `spectral_normalization`, `slack` and `soft_sampling`). Both were generated from commit `35dce05` with a clean tree.
