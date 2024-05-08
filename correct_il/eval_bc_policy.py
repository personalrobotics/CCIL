"""
Script to eval BC policies given data
"""
import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import argparse
import pickle
from matplotlib import pyplot as plt
import numpy as np
import d3rlpy
import d4rl

import torch
from utils import seed, parse_config, save_config_yaml, load_demo_for_policy, load_env, evaluate_on_environment, D3Agent
from envs import *

def construct_parser():
    parser = argparse.ArgumentParser(description='Training BC Policies.')
    parser.add_argument("config_path", help="Path to config file")
    return parser

def main():
    arg_parser = construct_parser()
    config = parse_config(arg_parser)
    output_folder = config.output.policy
    os.makedirs(output_folder, exist_ok=True)
    print(config)
    p_config = config.policy

    seed(config.seed)
    d3rlpy.seed(config.seed)
    env, meta_env = load_env(config)
    env.seed(config.seed)
    env.action_space.seed(config.seed)
    env.observation_space.seed(config.seed)

    # load
    device = 'cuda'
    policy = torch.jit.load(os.path.join(output_folder, f'policy.pt'))
    policy.to(device)
    agent = D3Agent(policy, device)

    sweep_noises = [0] if len(config.eval.noise) == 0 else config.eval.noise
    print(sweep_noises)
    color_begin='\033[92m'
    color_end='\033[0m'
    for noise in sweep_noises:
        rewards, success = evaluate_on_environment(env,agent,n_trials=100,metaworld=meta_env,
                                                   sensor_noise_size=noise, actuator_noise_size=noise)
        print(f'{color_begin}Noise size {noise} {np.average(rewards)} {np.std(rewards)}{color_end}')
        suffix = f"_noise_{noise}" if len(config.eval.noise) > 0 else ""
        pickle.dump(rewards,
            open(os.path.join(output_folder, f'rewards{suffix}.pkl'), 'wb'))
        pickle.dump(success/100,
            open(os.path.join(output_folder, f'success{suffix}.pkl'), 'wb'))

        with open('/tmp/eval_bc_policy.txt', 'a+') as f:
            f.write(f'{config.output.policy} {noise} {np.average(rewards)}')
            f.write('\n')

if __name__ == '__main__':
    main()
