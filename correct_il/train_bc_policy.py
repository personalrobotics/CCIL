"""
Script to train BC policies given data
"""

import argparse
import os
import d3rlpy
import numpy as np
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from d3rlpy.models.encoders import VectorEncoderFactory

from utils import seed, parse_config, save_config_yaml, load_demo_for_policy, dataset_to_d3rlpy, load_data
import CustomBC

def construct_parser():
    parser = argparse.ArgumentParser(description='Training BC Policies.')
    parser.add_argument("config_path", help="Path to config file")
    return parser

def exists_prev_output(output_folder, config):
    policy = os.path.join(output_folder, "policy.pt")
    return os.path.exists(policy)

def train_val_split(dataset: d3rlpy.dataset.MDPDataset, train_frac):
    n_data = len(dataset.observations)
    n_train = int(train_frac * n_data)
    n_val = n_data - n_train
    train_idx = np.random.choice(n_data, int(train_frac * n_data), replace=False)
    train_mask = np.zeros(n_data, dtype=bool)
    train_mask[train_idx] = True

    train_terminals = np.zeros(n_train)
    train_terminals[-1] = 1.0
    val_terminals = np.zeros(n_val)
    val_terminals[-1] = 1.0

    train_data = d3rlpy.dataset.MDPDataset(
        dataset.observations[train_mask],
        dataset.actions[train_mask],
        dataset.rewards[train_mask],
        train_terminals
    )
    val_data = d3rlpy.dataset.MDPDataset(
        dataset.observations[~train_mask],
        dataset.actions[~train_mask],
        dataset.rewards[~train_mask],
        val_terminals
    )
    return train_data, val_data

def main():
    arg_parser = construct_parser()
    config = parse_config(arg_parser)
    output_folder = config.output.policy
    os.makedirs(output_folder, exist_ok=True)
    p_config = config.policy

    if exists_prev_output(output_folder, config) and not config.overwrite:
        print(f"Found existing results in {output_folder}, quit")
        exit(0)

    seed(config.seed)
    d3rlpy.seed(config.seed)

    # Load Data
    full_dataset, expert_dataset, num_selected, num_total = load_demo_for_policy(config)

    # Create Agent
    agent = CustomBC.CustomBC(
        use_gpu=True,
        learning_rate=p_config.lr,
        batch_size=p_config.batch_size,
        policy_type='deterministic',
        encoder_factory=VectorEncoderFactory(hidden_units=p_config.layers) if p_config.layers else "default",
        scaler='standard',
        action_scaler='min_max',
        noise_cov=config.policy.noise_bc if config.policy.noise_bc else None)

    # Fit
    train_epochs = p_config.train_epochs
    if not p_config.naive and config.aug.type == 'noisy_action' and config.aug.balance_data:
        train_epochs //= config.aug.num_labels
    scorer = d3rlpy.metrics.scorer.continuous_action_diff_scorer
    agent.fit(full_dataset,
              n_epochs=train_epochs,
              save_interval=1e8,
              logdir=output_folder,
              verbose=False,
              show_progress=False,
              eval_episodes=val_dataset,
              scorers={
                'expert_data_error': scorer,
                }
              )

    # Save
    agent.save_policy(os.path.join(output_folder, f'policy.pt'))
    save_config_yaml(config, os.path.join(output_folder, f'config.yml'))

    import glob
    for f in sorted(glob.glob(os.path.join(output_folder, '*', '*.csv'))):
        if 'time_algorithm_update' not in f and 'time_sample_batch' not in f and 'time_step' not in f:
            f_basename = os.path.basename(f)
            os.rename(f, os.path.join(output_folder, f_basename))
        else:
            os.remove(f)


if __name__ == '__main__':
    main()
