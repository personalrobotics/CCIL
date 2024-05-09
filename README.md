# CCIL: Continuity-based Data Augmentation for Imitation Learning

Official repo for CCIL: Continuity-based Data Augmentation for Imitation Learning. This codebase allows you to
- Train dynamics functions from demonstration data while enforcing some form of local Lipschitz continuity
- Generate corrective labels using dynamics function
- Train behavior cloning agents from demonstrations or augmented dataset

## Setup

Tested with python `3.8.10`. Please note that we use our own forks of multiple packages:
- [d3rlpy](https://github.com/personalrobotics/d3rlpy/tree/release/CCIL) - fixed normalization bug when training behavioral cloning
- [gym-pybullet-drones](https://github.com/personalrobotics/gym-pybullet-drones/tree/fix-install) - added environments and fixed installation bugs

```bash
git clone https://github.com/personalrobotics/CCIL.git
cd CCIL
pip install -r install_deps.txt
pip install -e .
```

## Running experiments

We include scripts to reproduce our experiments:

```bash
> ./scripts/pendulum.sh
```

To visualize the results:

```bash
> find output/* -name "rewards_noise_*.pkl" | xargs python ./correct_il/tabulate_results.py
```

With additional environments installed (see below for instructions), you can try:
```bash
> ./scripts/mujoco.sh
> ./scripts/metaworld.sh
> ./scripts/drone.sh
> ./scripts/f1.sh
```

If you need to sweep over parameters, check out:
```bash
./scripts/sweep.sh
```

## Configuration Files

An experiment is parameterized by its configuration. Complete configuration files are given in `config/` for each task. Note that some fields, usually marked with `'...'`, are auto-populated when the config file is loaded with `utils.parse_config()`.

## Data Format

Data is stored in the `data` folder as a pickle file. Each data file is a list of trajectories,
where a trajectory is a dict with `"observations"` and `"actions"` keys. The respective corresponding
values are 2D ndarrays, where the first dimension is horizon and the second dimension is state
dimension and action dimension, respectively.

For example, if `T` is the length of a trajectory, `o` is the observation dimension, and `a` is the action dimension:

```
[
    {
        "observations": np.ndarray of size `T x o`,
        "actions": np.ndarray of size `T x a`
    },
    {
        "observations": np.ndarray of size `T x o`,
        "actions": np.ndarray of size `T x a`
    },
    ...
]
```

## CCIL Structure

End-to-end pipeline scripts are provided for each task suite in `scripts/`, but the overall sequential structure of CCIL is as follows:

1. Train a dynamics model from data. This is handled in `correct_il/train_dynamics_model.py`, which reads from the supplied config file the required parameters to train the dynamics model. Lipschitz regularization is applied as specified.
2. Augment data with corrective labels. This process happens in `correct_il/gen_aug_label.py`, again using parameters given in the config file, which dictate the label generation method and the corresponding parameters.
3. Train behavioral cloning on the augmented dataset, which is a standard BC algorithm. This is done in `correct_il/train_bc_policy.py`.
4. Evaluation is done in `corrective_il/eval_bc_policy.py`

## Training Baselines

Some baselines can be trained in this repository, namely vanilla BC and NoiseBC. Training vanilla bc is handled simply by overriding some parameters when training a BC model, replacing `<CONFIG PATH>` with the path to the config file:

```bash
python correct_il/train_bc_policy.py <CONFIG PATH> policy.naive true
```

Similarly, training NoiseBC is handled the same way, replacing `<NOISE>` with the appropriate noise magnitude. In our experiments we usually use `0.0001`:

``` bash
python correct_il/train_bc_policy.py <CONFIG PATH> policy.naive true policy.noise_bc <NOISE>
```
