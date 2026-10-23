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

from utils import D3Agent, seed, parse_config, load_trajectories, build_dataset, save_config_yaml, root_finder_s
from invariants import declared_action_bounds, transition_index, validate_trajectories
from models.nn_dynamics import checkpoint_name, load_world_model
from envs import *

def construct_parser():
    parser = argparse.ArgumentParser(description='Generate Augmentation Labels.')
    parser.add_argument("config_path", help="Path to config file")
    parser.add_argument("--test", action='store_true')
    return parser


def pack_data(s, a, sp, device):
    s_t = torch.as_tensor(s).float().to(device)
    a_t = torch.as_tensor(a).float().to(device)
    sp_t = torch.as_tensor(sp).float().to(device)
    return (s_t, a_t, sp_t)

# Every generator returns one record per attempted candidate, rejected ones
# included: arrays "s", "a", "s_next" (n, dim), "accepted" (n,) and "reason"
# (n,), where reason is None, "non_finite", "out_of_support", or
# "generation_failed". The support check is aug.epsilon under the rule
# load_demo_for_policy applies (distance < epsilon, skipped when epsilon is
# unset), so generation records the decision training already used.


def _candidates(gen_s, gen_a, next_s, origin, epsilon=None, failed=None):
    """Classify candidates against the declared validity checks.

    gen_s, gen_a, next_s: (n, dim) tensors. origin: the demonstration state
    the label perturbs. epsilon: aug.epsilon; no support check when it is
    falsy. failed: (n,) bool, True where the generator itself could not
    produce a candidate.
    """
    distance = torch.norm(gen_s - origin, dim=1)
    finite = torch.isfinite(gen_s).all(dim=1) & torch.isfinite(gen_a).all(dim=1) & torch.isfinite(origin).all()
    reasons = []
    for k in range(gen_s.shape[0]):
        if not finite[k]:
            reasons.append("non_finite")
        elif failed is not None and failed[k]:
            reasons.append("generation_failed")
        elif epsilon and not distance[k] < epsilon:
            reasons.append("out_of_support")
        else:
            reasons.append(None)
    return {
        "s": gen_s.detach().cpu().numpy(),
        "a": gen_a.detach().cpu().numpy(),
        "s_next": next_s.detach().cpu().numpy(),
        "accepted": np.array([r is None for r in reasons]),
        "reason": np.array(reasons, dtype=object),
        "distance": distance.detach().cpu().numpy(),
    }


def _root_find_one(loss, s, a, target, max_iter):
    """Root-find a single candidate; non-finite inputs skip the solver."""
    if not (torch.isfinite(s).all() and torch.isfinite(a).all() and torch.isfinite(target).all()):
        return torch.full_like(s, float("nan")), True
    s_prime, is_success, _ = root_finder_s(loss, s_next=target, a=a, init_s=s, max_iter=max_iter)
    return s_prime, not is_success


def gen_forward_euler(max_iter, epsilon, model, datapoint):
    # find s' such that s' + f(s', a) = s
    def loss(s, a, next_s):
        return s + model.f(s, a) - next_s

    s, a, sp = datapoint
    s_prime, failed = _root_find_one(loss, s, a, s, max_iter)  # TODO maybe change initial guess
    return _candidates(s_prime[None], a[None], s[None], s, epsilon=epsilon, failed=torch.tensor([failed]))


def gen_backward_euler(max_iter, epsilon, model, datapoint):
    # find s' such that s' + f(s, a) = s
    def loss(s, a, next_s):
        return s + model.f(next_s, a) - next_s

    s, a, sp = datapoint
    s_prime, failed = _root_find_one(loss, s, a, s, max_iter)
    return _candidates(s_prime[None], a[None], s[None], s, epsilon=epsilon, failed=torch.tensor([failed]))


def gen_backward_euler_fast(max_iter, epsilon, model, datapoint):
    # s' = s - f(s, a)
    with torch.no_grad():
        s, a, sp = datapoint
        gen_s = s[None] - model.f(s[None], a[None])
        return _candidates(gen_s, a[None], s[None], s, epsilon=epsilon)


def gen_noisy_action_be(num_labels, delta, epsilon, model, datapoint):
    # s' = s_next - f(s_next, a + noise)
    with torch.no_grad():
        s, a, sp = datapoint
        gen_noise = torch.normal(0, delta, size=(num_labels, a.size(0))).to(a.device)
        noisy_a = gen_noise + a
        next_s_tile = sp.repeat(num_labels, 1)
        gen_s = next_s_tile - model.f(next_s_tile, noisy_a)
        return _candidates(gen_s, noisy_a, next_s_tile, s, epsilon=epsilon)


def gen_noisy_action_fe(num_labels, delta, epsilon, max_iter, model, datapoint):
    # find s' such that s' + f(s', a + noise) = s_next
    def loss(s, a, next_s):
        return s + model.f(s, a) - next_s

    s, a, sp = datapoint
    gen_noise = torch.normal(0, delta, size=(num_labels, a.size(0))).to(a.device)
    noisy_a = a + gen_noise
    gen_s, failed = [], []
    for i in range(num_labels):
        s_prime, f = _root_find_one(loss, s, noisy_a[i], sp, max_iter)
        gen_s.append(s_prime)
        failed.append(f)
    return _candidates(torch.stack(gen_s), noisy_a, sp.repeat(num_labels, 1), s,
                       epsilon=epsilon, failed=torch.tensor(failed))


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
        return partial(gen_forward_euler, aug_config.max_iter, aug_config.epsilon)
    if aug_config.type == 'backward_euler':
        return partial(gen_backward_euler, aug_config.max_iter, aug_config.epsilon)
    if aug_config.type == 'backward_euler_fast' or aug_config.type == 'random_backward':
        return partial(gen_backward_euler_fast, aug_config.max_iter, aug_config.epsilon)
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
    return checkpoint_name(backward=aug_config.type == "backward_model")

def exists_prev_output(output_folder, config):
    f1 = os.path.join(output_folder, "aug_data.pkl")
    f2 = os.path.join(output_folder, "statistics.txt")
    return os.path.exists(f1) and os.path.exists(f2)

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

    # Load and validate data before the model, so malformed data fails first
    trajectories = load_trajectories(config.data)
    action_bounds = declared_action_bounds(config.data)
    validate_trajectories(trajectories, *(action_bounds or (None, None)))
    np_s, np_a, np_sp = build_dataset(trajectories)
    data_size, state_dim = np_s.shape
    _, action_dim = np_a.shape

    # Load Model
    model_basename = choose_model_basename(config.aug)
    model_fn = os.path.join(config.output.dynamics, model_basename)
    device = "cpu" if config.no_gpu else "cuda"
    model = load_world_model(model_fn, device)

    if config.aug.type == 'random_backward':
        # replace expert action with random action in data augmentation
        from utils import load_env
        env, _ = load_env(config)
        for i in range(data_size):
            np_a[i, :] = env.action_space.sample()

    s, a, sp = pack_data(np_s, np_a, np_sp, device)

    generator = choose_augmentation(config.aug)

    if config.debug and data_size > 5:
        data_size = 5

    labels = collect_labels(generator, model, (s, a, sp), np_s, data_size,
                            transition_ids=transition_index(trajectories), action_bounds=action_bounds)
    accepted = labels['accepted']
    counts = labels['counts']

    # aug_data.pkl holds accepted labels only. It is written even when no label is
    # valid, so a zero-label run leaves an explicit empty artifact instead of none.
    paths = {
        'observations': labels['s'][accepted].reshape(-1, state_dim),
        'actions': labels['a'][accepted].reshape(-1, action_dim),
        'next_obs': labels['s_next'][accepted].reshape(-1, state_dim),
        'original_states': labels['original_states'][accepted].reshape(-1, state_dim),
    }
    pickle.dump(paths,
        open(os.path.join(output_folder, f'aug_data.pkl'), 'wb'))
    save_config_yaml(config,
        os.path.join(output_folder, f'config.yml'))
    # Every attempted candidate with its decision; reason is '' for accepted labels.
    np.savez(os.path.join(output_folder, 'label_decisions.npz'),
             datapoint=labels['datapoint'], episode=labels['episode'], frame=labels['frame'],
             observations=labels['s'], actions=labels['a'],
             next_obs=labels['s_next'], original_states=labels['original_states'],
             distance=labels['distance'], accepted=accepted,
             reason=np.array(['' if r is None else r for r in labels['reason']]))
    if not accepted.any():
        print("No valid labels generated")

    distances = labels['distance'][np.isfinite(labels['distance'])]

    if config.aug.type == 'eval_noisy':
        with open(os.path.join(output_folder, f'statistics-{config.aug.delta}.txt'), 'w') as f:
            f.write(f'Avg errors: {np.average(distances)}')

        with open('/tmp/pendulum_eval_noisy.txt', 'a+') as f:
            f.write(f'{output_folder}-{config.aug.delta} {np.average(distances)}')
            f.write('\n')


    # Save statistics
    with open(os.path.join(output_folder, f'statistics.txt'), 'w') as f:
        f.write(f'Input Data Point: {data_size}\n')
        f.write(f'Attempted: {counts["attempted"]}\n')
        f.write(f'Accepted: {counts["accepted"]}\n')
        f.write(f'Rejected (non_finite): {counts["non_finite"]}\n')
        f.write(f'Rejected (out_of_support): {counts["out_of_support"]}\n')
        f.write(f'Generation failed: {counts["generation_failed"]}\n')
        f.write(f'Rejected (action_out_of_bounds): {counts["action_out_of_bounds"]}\n')
        if len(distances) > 0: f.write(f'Avg norm distance (finite candidates): {np.average(distances)}')
    if len(distances) > 0:
        pickle.dump({'distance': distances},
                open(os.path.join(output_folder, f'distance.pkl'), 'wb'))
        fig = plt.figure()
        ax = fig.add_subplot()
        fig.suptitle("Generated Label Distance")
        ax.hist(distances, density=True, bins=50)
        path = os.path.join(output_folder, "label_distance.png")
        fig.savefig(path)
    if config.debug:
        print(distances)


def collect_labels(generator, model, data, np_s, data_size, transition_ids=None, action_bounds=None):
    """Run the generator on the first data_size transitions.

    Returns every attempted candidate (rejected ones included) with its source
    datapoint, episode and frame, and the demonstration state it perturbs, plus
    reconciled counts. transition_ids: (episode, frame) arrays aligned with
    data. action_bounds: optional (low, high); accepted labels whose action
    falls outside are rejected as action_out_of_bounds, never clamped.
    """
    s, a, sp = data
    state_dim, act_dim = s.shape[1], a.shape[1]
    episode, frame = transition_ids if transition_ids is not None else (np.full(len(s), -1), np.full(len(s), -1))
    if action_bounds is not None:
        low = np.broadcast_to(np.asarray(action_bounds[0], dtype=float), (act_dim,))
        high = np.broadcast_to(np.asarray(action_bounds[1], dtype=float), (act_dim,))
    records = []
    for i in tqdm(range(data_size)):
        rec = generator(model, (s[i], a[i], sp[i]))
        n = len(rec['accepted'])
        # A malformed generator output is a programming defect, not a rejected label.
        expected = {'s': (n, state_dim), 'a': (n, act_dim), 's_next': (n, state_dim), 'reason': (n,), 'distance': (n,)}
        for k, shape in expected.items():
            if np.shape(rec[k]) != shape:
                raise ValueError(f"generator returned {k} with shape {np.shape(rec[k])}, expected {shape}")
        if action_bounds is not None:
            for k in range(n):
                if rec['accepted'][k] and ((rec['a'][k] < low) | (rec['a'][k] > high)).any():
                    rec['accepted'][k] = False
                    rec['reason'][k] = 'action_out_of_bounds'
        rec['datapoint'] = np.full(n, i)
        rec['episode'] = np.full(n, episode[i])
        rec['frame'] = np.full(n, frame[i])
        rec['original_states'] = np.tile(np_s[i], (n, 1))
        records.append(rec)
    keys = ('s', 'a', 's_next', 'accepted', 'reason', 'distance', 'datapoint', 'episode', 'frame', 'original_states')
    out = {k: np.concatenate([r[k] for r in records]) for k in keys}
    counts = {'attempted': len(out['reason']), 'accepted': int(out['accepted'].sum())}
    reasons = ('non_finite', 'out_of_support', 'generation_failed', 'action_out_of_bounds')
    for reason in reasons:
        counts[reason] = int(sum(r == reason for r in out['reason']))
    if counts['accepted'] + sum(counts[r] for r in reasons) != counts['attempted']:
        raise RuntimeError(f"label counts do not reconcile: {counts}")
    out['counts'] = counts
    return out

if __name__ == "__main__":
    main()
