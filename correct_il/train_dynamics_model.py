"""
Script to learn MDP model from data for offline policy optimization
"""

import argparse
import os
import pickle

from matplotlib import pyplot as plt
import numpy as np

from models.nn_dynamics import WorldModel
from utils import seed, parse_config, load_data, save_config_yaml


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


def save_loss(train_loss, folder_name, prediction_error, model_name=None, eval_loss=None):
    model_name = f"{model_name}_" if model_name else ""
    for loss_name, losses in train_loss.items():
      fn_prefix = os.path.join(folder_name, f'{model_name}train_{loss_name}')
      plot_loss(losses, fn_prefix + '.png', title=loss_name)
      with open(fn_prefix+'.txt', 'w') as f:
        _l = np.array2string(np.array(losses), formatter={'float_kind':lambda x: "%.6f\n" % x})
        f.write(_l)
    with open(os.path.join(folder_name, f'{model_name}statistics.txt'), 'w') as f:
      f.write(f'Avg Prediction Error (unnormalized) {prediction_error:.16f}')

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
    f1 = os.path.join(output_folder, "dynamics.pkl")
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
    s, a, sp = load_data(config.data)

    # Construct Dynamics Model
    d_config = config.dynamics
    dynamics = WorldModel(s.shape[1], a.shape[1], d_config=d_config,
                          hidden_size=d_config.layers,
                          fit_lr=d_config.lr,
                          fit_wd=d_config.weight_decay,
                          device="cpu" if config.no_gpu else "cuda",
                          activation=d_config.activation)

    # Fit Dynamics Model
    # learn forward dynamics
    train_loss = dynamics.fit_dynamics(
      s, a, sp,
      batch_size=d_config.batch_size,
      train_epochs=d_config.train_epochs,
      set_transformations=True)

    # Save Model and config
    save_config_yaml(config, os.path.join(output_folder, "config.yaml"))

    fn = "dynamics_backward.pkl" if d_config.backward else "dynamics.pkl"
    with open(os.path.join(output_folder, "dynamics.pkl"), "wb") as f:
        pickle.dump(dynamics, f)

    # Report Validation Loss
    prediction_error = dynamics.eval_prediction_error(s, a, sp, d_config.batch_size)

    # Save Training Loss
    save_loss(train_loss, output_folder, prediction_error, eval_loss=None)

    # Save distribution of local lipschitz coefficients over data
    local_L = dynamics.eval_lipschitz_coeff(s, a, batch_size=1024)
    lipschitz_constraint = d_config.lipschitz_constraint**(1 + len(d_config.layers))
    plot_lipschitz_dist(local_L, output_folder, lipschitz_constraint=lipschitz_constraint)

if __name__ == "__main__":
    main()
