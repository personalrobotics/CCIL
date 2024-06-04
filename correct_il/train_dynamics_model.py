"""
Script to learn MDP model from data for offline policy optimization
"""

import argparse
import os
import pickle

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from models.nn_dynamics import WorldModel
from utils import seed, parse_config, load_data, save_config_yaml

import math

# SMALL_SIZE = 5
# MEDIUM_SIZE = 6
# BIGGER_SIZE = 5
# plt.rc('font', size=SMALL_SIZE, family="Times New Roman")
# plt.rc('axes', titlesize=SMALL_SIZE)
# plt.rc('axes', labelsize=MEDIUM_SIZE)
# plt.rc('xtick', labelsize=SMALL_SIZE)
# plt.rc('ytick', labelsize=SMALL_SIZE)
# plt.rc('legend', fontsize=SMALL_SIZE)
# plt.rc('figure', titlesize=BIGGER_SIZE, figsize=(1.5, 0.75*1.5))

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
    max_local_L = np.max(lipschitz_coeff)
    fig.suptitle("Local Lipschitz Coefficient Distribution", x=0.62)
    ax.set_xlim(1, math.ceil(max_local_L * 2) / 2)
    ax.set_xticks([1, math.ceil(max_local_L * 2) / 2])
    ax.set_xlabel("Local Lipschitz Coefficient")
    # ax.set_xlabel("Local Lipschitz Coefficient", labelpad=-MEDIUM_SIZE)
    ax.set_ylabel("Proportion of Labels")
    ax.spines[['right', 'top']].set_visible(False)
    ax.hist(lipschitz_coeff, bins=50, weights=np.ones(len(lipschitz_coeff)) / len(lipschitz_coeff))
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(folder_name, f"{model_name}train_local_lipschitz.png"), dpi=300)
    fig.savefig(os.path.join(folder_name, f"{model_name}train_local_lipschitz.pdf"), format="pdf", transparent=True)

def plot_err_dist(err_norms, folder_name, model_name=None):
    model_name = f"{model_name}_" if model_name else ""
    fig = plt.figure()
    ax = fig.add_subplot()
    err_min, err_max = np.min(err_norms), np.max(err_norms)
    fig.suptitle(f"Model Error Distribution (min={err_min:.3f}, max={err_max:.3f})")
    ax.hist(err_norms, density=False, bins=50)
    path = os.path.join(folder_name, f"{model_name}train_err_dist.png")
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
    s, a, sp = load_data(config)

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
    pred_errs = dynamics.eval_prediction_error(s, a, sp, d_config.batch_size, reduce_err=False)

    # Save Training Loss
    save_loss(train_loss, output_folder, np.mean(pred_errs), eval_loss=None)
    plot_err_dist(pred_errs, output_folder)

    # Save distribution of local lipschitz coefficients over data
    local_L = dynamics.eval_lipschitz_coeff(s, a, batch_size=1024)
    if d_config.lipschitz_type != "none":
        if "spectral_normalization" in d_config.lipschitz_type:
            lipschitz_constraint = d_config.lipschitz_constraint**(1 + len(d_config.layers))
        else:
            lipschitz_constraint = d_config.lipschitz_constraint
    else:
        lipschitz_constraint = None
    # with open(os.path.join(output_folder, f"local_L_{lipschitz_constraint}.pkl"), "wb") as f:
    #     pickle.dump(local_L, f)
    plot_lipschitz_dist(local_L, output_folder, lipschitz_constraint=lipschitz_constraint)

if __name__ == "__main__":
    main()
