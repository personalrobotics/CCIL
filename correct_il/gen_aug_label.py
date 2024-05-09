"""
Script to generate corrective labels from demonstration using approximate dynamic function

forward_euler
    st - f(s', at) - s' = 0
backward_euler
    st - f(st, at) - s' = 0
noisy_action
    snext - f(s', at) - s' = 0
backward_model
    s' = f_backward(st, at)
"""

import argparse
import os
import pickle
import torch
from functools import partial
import numpy as np
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt
import math
from models.nn_dynamics import WorldModel

from utils import D3Agent, seed, parse_config, load_data, save_config_yaml, root_finder_s
from envs import *

SMALL_SIZE = 5
MEDIUM_SIZE = 6
BIGGER_SIZE = 7
plt.rc('font', size=SMALL_SIZE, family="Times New Roman")
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE, figsize=(1.5, 0.75*1.5))

def construct_parser():
    parser = argparse.ArgumentParser(description='Generate Augmentation Labels.')
    parser.add_argument("config_path", help="Path to config file")
    parser.add_argument("--test", action='store_true')
    return parser


def pack_data(device, *arrs):
    return [torch.as_tensor(arr).float().to(device) for arr in arrs]

@torch.no_grad()
def label_error(model: WorldModel, s, a, original_s, original_a, batch_size=None):
    is_batch = len(s.shape) > 1
    s = s.reshape(-1, s.shape[-1])
    a = a.reshape(-1, a.shape[-1])
    original_s = original_s.reshape(-1, original_s.shape[-1])
    original_a = original_a.reshape(-1, original_a.shape[-1])

    batch_size = batch_size if batch_size else len(s)
    local_L = []
    for i in range(0, len(s), batch_size):
        batch_os, batch_oa = pack_data('cuda', original_s[i:i+batch_size], original_a[i:i+batch_size])
        L = model.local_lipschitz_coeff(batch_os, batch_oa)
        local_L.append(L.detach().cpu().numpy())
    local_L = np.concatenate(local_L, axis=0)
    sa = np.concatenate([s, a], axis=-1)
    original_sa = np.concatenate([original_s, original_a], axis=-1)
    dist = np.linalg.norm(original_sa - sa, axis=-1)
    approx_err = local_L * dist
    if not is_batch:
        dist = dist[0]
        local_L = local_L[0]
        approx_err = approx_err[0]
    return {
        "distance": dist,
        "local_L": local_L,
        "label_err": approx_err
    }

# TODO fix the count_fail
def gen_forward_euler(max_iter, model: WorldModel, datapoint):
    # given s, a, sp
    # find s_prime that s_prime + (s_prime, a) -> s
    # s_prime + f(s_prime,a) - s = 0

    def loss(s, a, next_s):
        return s + model.f(s, a) - next_s

    s, a, sp = datapoint
    with torch.no_grad():
        s_g = s
        for _ in range(max_iter):
            s_g = s - model.f(s_g, a)
        is_success = torch.linalg.norm(s_g + model.f(s_g, a) - s).item() <= 1e-4
    s_prime = s_g
    # s_prime, is_success, count_fail = root_finder_s(
    #     loss, s_next=s, a=a, init_s=s, max_iter=max_iter) # TODO maybe change initial guess
    if is_success:
        gen_data = (s_prime.cpu().numpy(), a.cpu().numpy(), s.cpu().numpy())
        distance = np.linalg.norm(s.cpu().numpy() - s_prime.cpu().numpy())
        info = {'distance': [distance]}
    else:
        gen_data = (None, None, None)
        info = None
    return int(is_success), int(not is_success), gen_data, info

def gen_backward_euler(max_iter, model, datapoint):
    # given s, a, sp
    # find s_prime that s_prime + (s, a) -> s

    def loss(s, a, next_s):
        return s + model.f(next_s, a) - next_s

    s, a, sp = datapoint
    s_prime, is_success, count_fail = root_finder_s(
        loss, s_next=s, a=a, init_s=s, max_iter=max_iter)
    if is_success:
        gen_data = (s_prime.cpu().numpy(), a.cpu().numpy(), s.cpu().numpy())
        distance = np.linalg.norm(s.cpu().numpy() - s_prime.cpu().numpy())
    else:
        gen_data = (None, None, None)
        distance = 0

    if count_fail >= max_iter:
        count_fail = 1
    else:
        count_fail = 0
    return int(is_success), count_fail, gen_data, {'distance':[distance]}

def gen_backward_euler_fast(max_iter, model, datapoint):
    # s_prime = s - f(s,a)
    with torch.no_grad():
        s, a, sp = datapoint
        gen_s = torch.unsqueeze(s, 0) - model.f(torch.unsqueeze(s, 0), torch.unsqueeze(a, 0))
        gen_s = gen_s[0]
        distance = torch.norm(gen_s - s)

        count_success, count_fail = 1, 0

        gen_s = gen_s.cpu().numpy()

        gen_a = a.cpu().numpy()
        next_s_np = s.cpu().numpy()
        distance = distance.cpu().numpy()

        gen_data = (gen_s, gen_a, next_s_np)
        return count_success, count_fail, gen_data, {'distance':[distance]}

def gen_noisy_action_be(num_labels, delta, epsilon, model, datapoint):
    # return s + model.f(next_s, a_noisy) - next_s = 0
    # return next_s - model.f(next_s, a_noisy)

    with torch.no_grad():
        s, a, sp = datapoint
        gen_noise = torch.normal(0, delta, size=(num_labels, a.size(0))).to(a.device)
        noisy_a = gen_noise + a
        next_s_tile = sp.repeat(num_labels, 1)
        gen_s = next_s_tile - model.f(next_s_tile, noisy_a)
        distance = torch.norm(gen_s - s, dim=1)

        count_fail =  torch.count_nonzero(distance > epsilon).cpu().item()
        count_success = num_labels # - count_fail

        gen_s = gen_s.cpu().numpy()
        gen_a = noisy_a.cpu().numpy()
        next_s_np = next_s_tile.cpu().numpy()
        distance = distance.cpu().numpy()

        gen_data = (gen_s, gen_a, next_s_np)
        return count_success, count_fail, gen_data, {'distance':distance}

def gen_noisy_action_fe(num_labels, delta, epsilon, max_iter, model, datapoint):

    def loss(s, a, next_s):
        return s + model.f(s, a) - next_s

    s, a, sp = datapoint
    gen_noise = torch.normal(0, delta, size=(num_labels, a.size(0))).to(a.device)
    noisy_a = a + gen_noise
    count_success = 0
    count_fail = 0
    gen_s = []
    gen_a = []
    for i in range(num_labels):
        _a = noisy_a[i]
        s_prime, is_success, num_fail = root_finder_s(
            loss, s_next=sp, a=_a, init_s=s, max_iter=max_iter)
        if is_success and torch.norm(s_prime - s) <= epsilon:
            count_success += 1
            gen_s.append(s_prime.cpu().numpy())
            gen_a.append(_a.cpu().numpy())
        else:
            count_fail += 1
    sp = sp.cpu().numpy()
    gen_data = (gen_s, gen_a, sp.repeat(len(gen_s)))
    return count_success, count_fail, gen_data, {}

def gen_backward_model(model, datapoint):
    s, a, sp = datapoint
    prev_s = s + model.f(s, a)
    data = (prev_s.detach().cpu().numpy(), a.detach().cpu().numpy(), s.detach().cpu().numpy())
    return prev_s.shape[0], 0, data

def gen_backward_expert(expert: D3Agent, max_iter, model, datapoint):
    success, fail, data = gen_backward_euler(max_iter, model, datapoint)
    if not success:
        return success, fail, data

    s, a, sp = data
    with torch.no_grad():
        s_torch = torch.as_tensor(s).to(expert.device)
        expert_a = expert.policy(s_torch).cpu().numpy().reshape(a.shape)
        next_state = model.predict(s, expert_a).cpu().numpy().reshape(sp.shape)
    return 1, 0, (s, expert_a, next_state)

def choose_augmentation(aug_config):
    if aug_config.type == 'forward_euler':
        return partial(gen_forward_euler, aug_config.max_iter)
    if aug_config.type == 'backward_euler':
        return partial(gen_backward_euler, aug_config.max_iter)
    if aug_config.type == 'backward_euler_fast' or aug_config.type == 'random_backward':
        return partial(gen_backward_euler_fast, aug_config.max_iter)
    if aug_config.type == 'noisy_action':
        return partial(gen_noisy_action_be,
            aug_config.num_labels, aug_config.delta, aug_config.epsilon)
    if aug_config.type == 'noisy_action_fe':
        return partial(gen_noisy_action_fe,
            aug_config.num_labels, aug_config.delta, aug_config.epsilon,
            aug_config.max_iter)
    if aug_config.type == "backward_model":
        return gen_backward_model
    if aug_config.type == "backward_expert":
        expert = torch.jit.load("output/expert_policy/pendulum.pt")
        expert.to("cuda")
        agent = D3Agent(expert, "cuda")
        return partial(gen_backward_expert, agent, aug_config.max_iter)
    raise Exception("Unknown type of augmentation")

def choose_model_basename(aug_config):
    if aug_config.type == "backward_model":
        return "dynamics_backward.pkl"
    else:
        return "dynamics.pkl"

def exists_prev_output(output_folder, config):
    f1 = os.path.join(output_folder, "aug_data.pkl")
    f2 = os.path.join(output_folder, "statistics.txt")
    return os.path.exists(f1) and os.path.exists(f2)

def model_fits_data(model, s, a, sp, thresh):
    if thresh:
        sp_pred = s + model.f(s.unsqueeze(0), a.unsqueeze(0))[0]
        return torch.linalg.norm(sp_pred - sp).item() < thresh
    return True

def main():
    arg_parser = construct_parser()
    config = parse_config(arg_parser)
    output_folder = config.output.aug
    os.makedirs(output_folder, exist_ok=True)
    print(config)

    if exists_prev_output(output_folder, config) and not config.overwrite:
        print(f"Found existing results in {output_folder}, quit")
        exit(0)

    seed(config.seed)

    # Load Model
    model_basename = choose_model_basename(config.aug)
    model_fn = os.path.join(config.output.dynamics, model_basename)
    model: WorldModel = pickle.load(open(model_fn, 'rb'))

    # Load Data
    np_s, np_a, np_sp = load_data(config.data)
    data_size, state_dim = np_s.shape
    _, action_dim = np_a.shape

    if config.aug.type == 'random_backward':
        # replace expert action with random action in data augmentation
        from utils import load_env
        env, _ = load_env(config)
        for i in range(data_size):
            np_a[i, :] = env.action_space.sample()

    s, a, sp = pack_data('cuda', np_s, np_a, np_sp)

    generator = choose_augmentation(config.aug)

    original_states = []
    original_actions = []
    new_states = []
    new_actions = []
    new_next_states = []
    count_success = 0
    count_fail = 0

    if config.debug and data_size > 5:
        data_size = 5

    distances = []
    for i in tqdm(range(data_size)):
        if model_fits_data(model, s[i], a[i], sp[i], config.aug.model_err_thresh):
            _count_suc, _count_fail, gen_data, info = generator(model, (s[i], a[i], sp[i]))
        else:
            _count_suc, _count_fail, gen_data, info = 0, 1, None, None
        if info is not None: distances.append(info['distance'])
        if _count_suc > 0:
            count_success += _count_suc
            s_prime, a_prime, s_next = gen_data
            original_states.append(np.tile(np_s[i], (_count_suc, 1)))
            original_actions.append(np.tile(np_a[i], (_count_suc, 1)))
            new_states.append(s_prime)
            new_actions.append(a_prime)
            new_next_states.append(s_next)
        count_fail += _count_fail

    print(f"Generated {count_success} new labels with {count_fail} failures")

    if count_success > 0:
        original_states = np.concatenate(original_states, axis=0).reshape(-1, state_dim)
        original_actions = np.concatenate(original_actions, axis=0).reshape(-1, action_dim)
        new_states = np.concatenate(new_states, axis=0).reshape(-1, state_dim)
        new_actions = np.concatenate(new_actions, axis=0).reshape(-1, action_dim)
        new_next_states = np.concatenate(new_next_states, axis=0).reshape(-1, state_dim)

        # Save Data
        paths = {
            'observations': new_states,
            'actions': new_actions,
            'next_obs': new_next_states,
            'original_states': original_states,
            'original_actions': original_actions
        }
        pickle.dump(paths,
            open(os.path.join(output_folder, f'aug_data.pkl'), 'wb'))
        save_config_yaml(config,
            os.path.join(output_folder, f'config.yml'))
    else:
        print("No data generated")

    if config.aug.type == 'eval_noisy':
        with open(os.path.join(output_folder, f'statistics-{config.aug.delta}.txt'), 'w') as f:
            f.write(f'Avg errors: {np.average(np.concatenate(distances))}')

        with open('/tmp/pendulum_eval_noisy.txt', 'a+') as f:
            f.write(f'{output_folder}-{config.aug.delta} {np.average(np.concatenate(distances))}')
            f.write('\n')


    # Save statistics
    with open(os.path.join(output_folder, f'statistics.txt'), 'w') as f:
        f.write(f'Input Data Point: {data_size}\n')
        f.write(f'Success: {count_success}\n')
        f.write(f'Failure: {count_fail}\n')
        if len(distances) > 0: f.write(f'Avg norm distance: {np.average(np.concatenate(distances))}')
    if len(distances) > 0 and count_success > 0:
        pickle.dump({'distance':distances},
                open(os.path.join(output_folder, f'distance.pkl'), 'wb'))
        cat_distances = np.concatenate(distances)
        fig = plt.figure()
        ax = fig.add_subplot()
        fig.suptitle("Generated Label Distance")
        ax.hist(cat_distances, bins=100)
        if config.aug.epsilon:
            ax.axvline(config.aug.epsilon, color="black", linestyle="--")
        path = os.path.join(output_folder, "label_distance.png")
        fig.savefig(path)

        err_info = label_error(model, new_states, new_actions, original_states, original_actions, 1024)
        with open(os.path.join(output_folder, "label_err.pkl"), "wb") as f:
            pickle.dump(err_info, f)
        fig = plt.figure()
        ax = fig.add_subplot()
        fig.suptitle("Generated Label Error")
        ax.hist(err_info["label_err"], bins=100)
        fig.savefig(os.path.join(output_folder, "label_err.png"))

        fig = plt.figure()
        ax = fig.add_subplot()
        fig.suptitle("Generated Label Error CDF")
        max_error = np.max(err_info["label_err"])
        ax.set_xlim(0, math.ceil(max_error * 1000) / 1000)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, math.ceil(max_error * 1000) / 1000])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Error", labelpad=-MEDIUM_SIZE)
        ax.set_ylabel("Proportion of Labels \n (Cumulative)", labelpad=-MEDIUM_SIZE)
        ax.spines[['right', 'top']].set_visible(False)
        hist, bins, _ = ax.hist(err_info["label_err"], bins=10000, histtype="step", cumulative=True, density=True, alpha=0)
        ax.plot(bins[:-1], hist, color="tab:blue")
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(output_folder, "label_err_cdf.png"), dpi=300)
        fig.savefig(os.path.join(output_folder, "label_err_cdf.pdf"), format="pdf", transparent=True)

    if config.debug:
        print(distances)
        #import pdb;pdb.set_trace()

if __name__ == "__main__":
    main()
