# Legacy harness

`stubs/` holds minimal stand-ins for `gym` and `d4rl`. They let the repository's
own entry points (`correct_il/*.py`) be imported and run for workflows that never
construct a simulator, such as dynamics training and label generation on
Pendulum data, without installing the retired legacy stack (gym 0.19,
mujoco-py, D4RL). See the migration contract, section 3.

    PYTHONPATH=<repo>:<repo>/tools/legacy_harness/stubs python correct_il/train_dynamics_model.py ...

Any code path that calls `gym.make` fails with an AttributeError instead of
silently running a fake environment.
