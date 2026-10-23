"""
Script to learn MDP model from data for offline policy optimization
"""

import argparse
import json
import os

from matplotlib import pyplot as plt
import numpy as np

from models.nn_dynamics import WorldModel, checkpoint_name, save_world_model
from utils import seed, parse_config, load_trajectories, build_dataset, save_config_yaml
from validation import held_out_evaluation, load_or_create_split
from invariants import declared_action_bounds, validate_trajectories


def construct_parser():
    parser = argparse.ArgumentParser(description='Training Dynamic Functions.')
    parser.add_argument("config_path", help="Path to config file")
    return parser


def plot_loss(train_loss, fn, xbar=None, title='Train Loss'):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Dynamics Model Loss")
    ax.set_ylabel(title)
    ax.set_xlabel("Epoch")
    if xbar:
        ax.axhline(xbar, linestyle="--", color="black")
    ax.plot(train_loss)
    fig.savefig(fn)


def save_loss(train_loss, folder_name, train_error, held_out=None, model_name=None):
    model_name = f"{model_name}_" if model_name else ""
    for loss_name, losses in train_loss.items():
      fn_prefix = os.path.join(folder_name, f'{model_name}train_{loss_name}')
      plot_loss(losses, fn_prefix + '.png', title=loss_name)
      with open(fn_prefix+'.txt', 'w') as f:
        _l = np.array2string(np.array(losses), formatter={'float_kind':lambda x: "%.6f\n" % x})
        f.write(_l)
    with open(os.path.join(folder_name, f'{model_name}statistics.txt'), 'w') as f:
      f.write(f'Training-set one-step L2 error, label model (not a held-out metric): {train_error:.16f}\n')
      if held_out is None:
        f.write('Held-out one-step L2 error: not computed (dynamics.validation_fraction is 0)\n')
      else:
        agg = held_out['aggregate']
        f.write(f"Held-out one-step L2 error, evaluation model ({agg['trajectories']} trajectories, "
                f"{agg['transitions']} transitions): pooled {agg['pooled_mean_l2']:.16f}, "
                f"per-trajectory mean {agg['macro_mean_l2']:.16f}; see validation.json\n")

def plot_lipschitz_dist(lipschitz_coeff, folder_name, model_name=None, lipschitz_constraint=None):
    model_name = f"{model_name}_" if model_name else ""
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.suptitle("Local Lipschitz Coefficient")
    ax.hist(lipschitz_coeff, density=True, bins=50)
    if lipschitz_constraint:
        ax.axvline(lipschitz_constraint, linestyle="--", color="black")
    path = os.path.join(folder_name, f"{model_name}train_local_lipschitz.png")
    fig.savefig(path)

def exists_prev_output(output_folder, config):
    f1 = os.path.join(output_folder, checkpoint_name(config.dynamics.backward))
    f2 = os.path.join(output_folder, "statistics.txt")
    return os.path.exists(f1) and os.path.exists(f2)

def main():
    arg_parser = construct_parser()
    config = parse_config(arg_parser)
    output_folder = config.output.dynamics
    os.makedirs(output_folder, exist_ok=True)
    print(config)

    if exists_prev_output(output_folder, config) and not config.overwrite:
        print(f"Found existing results in {output_folder}, quit")
        exit(0)

    seed(config.seed)

    # Load Data
    d_config = config.dynamics
    trajectories = load_trajectories(config.data)
    validate_trajectories(trajectories, *(declared_action_bounds(config.data) or (None, None)))
    # Decide the held-out split before any training, so an impossible split fails fast.
    validation_fraction = d_config.validation_fraction or 0
    split = None
    if validation_fraction:
        split, reused = load_or_create_split(os.path.join(output_folder, "split.json"),
                                             len(trajectories), validation_fraction, config.seed)
        print(f"{'Reusing' if reused else 'Created'} split: {len(split['train'])} train, "
              f"{len(split['validation'])} validation trajectories")

    def fit(trajs):
        s, a, sp = build_dataset(trajs)
        if d_config.backward:
            # A backward model maps (s_{t+1}, a_t) to s_t, as gen_backward_model assumes.
            s, sp = sp, s
        model = WorldModel(s.shape[1], a.shape[1], d_config=d_config,
                           hidden_size=d_config.layers,
                           fit_lr=d_config.lr,
                           fit_wd=d_config.weight_decay,
                           device="cpu" if config.no_gpu else "cuda",
                           activation=d_config.activation)
        loss = model.fit_dynamics(
          s, a, sp,
          batch_size=d_config.batch_size,
          train_epochs=d_config.train_epochs,
          set_transformations=True)
        return model, loss, (s, a, sp)

    # Fit the label model on every trajectory
    dynamics, train_loss, (s, a, sp) = fit(trajectories)

    # Save Model and config
    save_config_yaml(config, os.path.join(output_folder, "config.yaml"))

    save_world_model(dynamics, os.path.join(output_folder, checkpoint_name(d_config.backward)), d_config)

    # Training-set error of the label model; held-out error comes from a separate fit
    train_error = dynamics.eval_prediction_error(s, a, sp, d_config.batch_size)
    held_out = None
    if split is not None:
        seed(config.seed)
        report = held_out_evaluation(trajectories, split, lambda t: fit(t)[0], backward=bool(d_config.backward))
        report["split_file"] = "split.json"
        with open(os.path.join(output_folder, "validation.json"), "w") as f:
            json.dump(report, f, indent=1)
        held_out = report["held_out"]

    # Save Training Loss
    save_loss(train_loss, output_folder, train_error, held_out)

    # Save distribution of local lipschitz coefficients over data
    local_L = dynamics.eval_lipschitz_coeff(s, a, batch_size=1024)
    lipschitz_constraint = d_config.lipschitz_constraint**(1 + len(d_config.layers))
    plot_lipschitz_dist(local_L, output_folder, lipschitz_constraint=lipschitz_constraint)

if __name__ == "__main__":
    main()
