#################################################################
# Utils
#################################################################
import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import torch
import numpy as np
import random
from typing import Tuple
import pickle
import gym,d4rl


from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from envs import *


def seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(trajectories: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.concatenate([traj["observations"][:-1] for traj in trajectories])
    a = np.concatenate([traj["actions"][:-1] for traj in trajectories])
    if len(a.shape) == 1:
        a = np.expand_dims(a, axis=-1)
    sp = np.concatenate([traj["observations"][1:] for traj in trajectories])
    return s, a, sp

def load_data(config_data):
    # TODO limit data number
    data_path = config_data.pkl
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return build_dataset(data)

def dataset_to_d3rlpy(s, a, _):
    import d3rlpy
    rews = np.ones(len(s))
    terminals = np.zeros(len(s)) # Hack
    terminals[-1] = 1.0
    return d3rlpy.dataset.MDPDataset(
        observations=np.array(s, dtype=np.float32),
        actions=np.array(a, dtype=np.float32),
        rewards=np.array(rews, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
        episode_terminals=np.array(terminals, dtype=np.float32),
    )

def gen_noise_dataset(obs, noise):
    return obs + np.random.normal(0,noise,obs.shape)

def load_demo_for_policy(config):
    # Load Expert Data
    expert_dataset = dataset_to_d3rlpy(*load_data(config.data))
    if config.policy.naive:
        return expert_dataset, expert_dataset, 0, 0

    # if config.policy.noise_bc:
    #     full_dataset = dataset_to_d3rlpy(*load_data(config.data))
    #     s, a, sp = load_data(config.data)
    #     for _ in range(config.aug.num_labels):
    #         # import pdb;pdb.set_trace()
    #         new_s = gen_noise_dataset(s,0.0001)
    #         noise_dataset = dataset_to_d3rlpy(new_s,a,sp)

    #         full_dataset.extend(noise_dataset)
    #         full_dataset.extend(expert_dataset)

    #     return full_dataset, expert_dataset

    # Load Aug Data
    aug_pkl_fn = os.path.join(config.output.aug, 'aug_data.pkl')
    aug_data = pickle.load(open(aug_pkl_fn, 'rb'))
    _s = aug_data['observations']
    _original_s = aug_data['original_states']

    sel_bounded = np.ones(len(_s), dtype=bool)
    if config.aug.label_err_quantile or config.aug.max_label_err:
        label_err_pkl = os.path.join(config.output.aug, "label_err.pkl")
        with open(label_err_pkl, "rb") as f:
            info = pickle.load(f)
        if config.aug.label_err_quantile:
            thresh = np.quantile(info["label_err"], config.aug.label_err_quantile)
            print(f'\033[93m Choosing label error threshold: {thresh:.3g} \033[0m')
            sel_bounded &= info["label_err"] <= thresh
        if config.aug.max_label_err:
            sel_bounded &= info["label_err"] <= config.aug.max_label_err
    if config.aug.epsilon:
        sel_bounded &= np.linalg.norm(_s - _original_s, axis=1) < config.aug.epsilon

    print(f'\033[93m Selected {sel_bounded.sum()} data points out of {len(_s)} \033[0m')
    aug_dataset = dataset_to_d3rlpy(
        aug_data['observations'][sel_bounded],
        aug_data['actions'][sel_bounded],
        aug_data['next_obs'][sel_bounded])



    # def batch_forward(f, states, actions):
    #     return [f(s.to('cpu').detach().numpy(),a.to('cpu').detach().numpy()) for s,a in zip(states, actions)]

    # env = gym.make('PendulumSwingupCont-v0')

    # accm_errors = []
    # for s,a,sp in zip(aug_data['observations'][sel_bounded],aug_data['actions'][sel_bounded],aug_data['next_obs'][sel_bounded]):
    #     gt_next = env.forward_solver(s,a)
    #     errors = sp - np.array(gt_next)
    #     errors = np.linalg.norm(errors)
    #     accm_errors.append(errors)

    # print(np.average(accm_errors))
    # import pdb;pdb.set_trace()
    # exit()

    aug_dataset.extend(expert_dataset)
    if config.aug.type == 'noisy_action' and config.aug.balance_data:
        for _ in range(config.aug.num_labels):
            aug_dataset.extend(expert_dataset)

    # Last two are the number of selected data points and the total number of data points
    return aug_dataset, expert_dataset, sel_bounded.sum(), len(_s)

def load_env(config):

    meta_env = False
    if config.env in ['hover-aviary-v0', 'flythrugate-aviary-v0', 'circle-aviary-v0']:
        import gym_pybullet_drones
        from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
        DEFAULT_OBS = ObservationType('kin')
        DEFAULT_ACT = ActionType('rpm')
        env = gym.make(f'{config.env}',
                            aggregate_phy_steps=1,
                            obs=DEFAULT_OBS,
                            act=DEFAULT_ACT,
                            gui=False
                            )
    elif config.env in ['assembly-v2', 'bin-picking-v2', 'button-press-topdown-v2', 'door-open-v2', 'drawer-close-v2', 'drawer-open-v2','handle-press-v2','plate-slide-v2',\
    'window-open-v2','disassemble-v2','faucet-open-v2','faucet-close-v2','coffee-button-v2','peg-unplug-side-v2','pick-place-wall-v2','reach-wall-v2',\
    'basketball-v2','coffee-pull-v2','coffee-push-v2','lever-pull-v2','pick-place-v2','push-v2','push-back-v2','reach-v2','soccer-v2','sweep-v2','window-close-v2']:

        import metaworld
        import metaworld.envs.mujoco.env_dict as _env_dict
        env = _env_dict.MT50_V2[config.env]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        meta_env = True
    else:
        env = gym.make(config.env)
    return env, meta_env

def evaluate_on_environment(
    env: gym.Env, algo, n_trials: int = 10, render: bool = False, metaworld=False,
    sensor_noise_size = None, actuator_noise_size=None):
    """Returns scorer function of evaluation on environment.
    """
    episode_rewards = []
    success = 0
    trail = 0
    drone_task = True if (env.spec is not None) and ('aviary' in env.spec.id) else False
    while True:
        observation = env.reset()
        if drone_task:
            observation = env.getDroneStateVector(0)
        episode_reward = 0.0
        steps = 0
        while True:

            # take action
            if sensor_noise_size:
                observation = observation * (1-sensor_noise_size) + sensor_noise_size * \
                    env.observation_space.sample() if not drone_task else \
                    np.random.uniform(low=-1.0, high=1.0, size=observation.shape)
            action = algo.predict([observation]) # no rescaling necessary
            if actuator_noise_size:
                action = action * (1-actuator_noise_size) + env.action_space.sample() * actuator_noise_size
            # action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            if drone_task:
                observation = env.getDroneStateVector(0)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
            if metaworld and steps > 499:
                break
            steps += 1
        success += info['success'] if 'success' in info else 0
        episode_rewards.append(episode_reward)
        trail+=1
        if trail >=n_trials:
            break

    return episode_rewards, success

class D3Agent():
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def load(self, model_folder, device):
        # load is handled at init
        pass
    # For 1-batch query only!
    def predict(self, sample):
        with torch.no_grad():
            input = torch.from_numpy(sample[0]).float().unsqueeze(0).to(self.device)
            at = self.policy(input)[0].cpu().numpy()
        return at

# ---------------------------------------------------------------
# Calculate EMD
# ---------------------------------------------------------------

def get_emd(X,Y) -> float:
    # get EMD/linear assignment
    #The points are arranged as m n-dimensional row vectors in the matrix X.
    d = cdist(X, Y)
    if np.inf in d:
        import pdb;pdb.set_trace()

    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / X.shape[0]


# ---------------------------------------------------------------
# Find root for equations
#

def root_finder_s(loss, s_next, a, init_s,
        max_iter=20,
        threshold=1e-4):
    # Find s as a solution to loss(s,a,s_next)
    # return s, is_success, # of iterations to solve

    s = init_s

    for i in range(max_iter):

        with torch.no_grad():
            r = loss(s, a, s_next)
            if torch.norm(r) <= threshold:
                break

        jac_s, _, _ = torch.autograd.functional.jacobian(
            loss, (s, a, s_next))

        delta_s = np.linalg.solve(
            jac_s.detach().cpu().numpy(),
            r.detach().cpu().numpy())
        s = s - torch.as_tensor(delta_s).to(s.device)

    if torch.norm(r) <= threshold:
        return s.detach(), 1, i

    return s.detach(), 0, max_iter


# ---------------------------------------------------------------
# Config Parser
# ---------------------------------------------------------------

import collections
import yaml
import os

def parse_config(parser):
    args, overrides = parser.parse_known_args()
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config.update(vars(args))

    if not hasattr(config, 'debug'):
        config['debug'] = False

    if not hasattr(config, 'overwrite'):
        config['overwrite'] = 1

    config = parse_overrides(config, overrides)

    config['output']['location'] = os.path.expandvars(config['output']['location'])
    config['output']['location'] = os.path.join(config['output']['location'], f'seed{config["seed"]}')
    config['data']['pkl'] = os.path.expandvars(config['data']['pkl'])
    config['output']['folder'] = os.path.join(
        config['output']['location'],
        f"{config['env']}{config['output']['folder_suffix']}")
    _folder = config['output']['folder']
    config = Namespace(config)

    print(config.policy.naive, "??")

    # Generate intermediate filenames
    dynamics_optional = "" if config.dynamics.lipschitz_type == "none" else f"L{config.dynamics.lipschitz_constraint}"
    config['output']['dynamics'] = os.path.join(_folder, f"dynamics_{config.dynamics.lipschitz_type}{dynamics_optional}")
    config['output']['aug'] = os.path.join(_folder, f"data/{config.aug.type}_{config.dynamics.lipschitz_type}{dynamics_optional}")
    if config.policy.naive:
        policy_folder = "policy/naive"
    else:
        policy_folder = f"policy/{config.aug.type}_{config.dynamics.lipschitz_type}{dynamics_optional}"
    if config.policy.noise_bc:
        # policy_folder = f"{policy_folder}_noise_{config.policy.noise_bc}"
        policy_folder = f"policy/noise_{config.policy.noise_bc}"
    config['output']['policy'] = os.path.join(_folder, policy_folder)
    return config


def save_config_yaml(config, save_fn):
    data = dict(config)
    with open(save_fn, 'w') as out_file:
        yaml.dump(data, out_file, default_flow_style=False)


def parse_overrides(config, overrides):
    """
    Overrides the values specified in the config with values.
    config: (Nested) dictionary of parameters
    overrides: Parameters to override and new values to assign. Nested
        parameters are specified via dot notation.
    >>> parse_overrides({}, [])
    {}
    >>> parse_overrides({}, ['a'])
    Traceback (most recent call last):
      ...
    ValueError: invalid override list
    >>> parse_overrides({'a': 1}, [])
    {'a': 1}
    >>> parse_overrides({'a': 1}, ['a', 2])
    {'a': 2}
    >>> parse_overrides({'a': 1}, ['b', 2])
    Traceback (most recent call last):
      ...
    KeyError: 'b'
    >>> parse_overrides({'a': 0.5}, ['a', 'test'])
    Traceback (most recent call last):
      ...
    ValueError: could not convert string to float: 'test'
    >>> parse_overrides(
    ...    {'a': {'b': 1, 'c': 1.2}, 'd': 3},
    ...    ['d', 1, 'a.b', 3, 'a.c', 5])
    {'a': {'b': 3, 'c': 5.0}, 'd': 1}
    """
    if len(overrides) % 2 != 0:
        # print('Overrides must be of the form [PARAM VALUE]*:', ' '.join(overrides))
        raise ValueError('invalid override list')

    for param, value in zip(overrides[::2], overrides[1::2]):
        keys = param.split('.')
        params = config
        for k in keys[:-1]:
            if k not in params:
                raise KeyError(param)
            params = params[k]
        if keys[-1] not in params:
            raise KeyError(param)

        current_type = type(params[keys[-1]])
        value = current_type(value)  # cast to existing type
        params[keys[-1]] = value

    return config


# Namespace for command line overwriting of training
class Namespace(collections.abc.MutableMapping):
    """Utility class to convert a (nested) dictionary into a (nested) namespace.
    >>> x = Namespace({'foo': 1, 'bar': 2})
    >>> x.foo
    1
    >>> x.bar
    2
    >>> x.baz
    Traceback (most recent call last):
        ...
    KeyError: 'baz'
    >>> x
    {'foo': 1, 'bar': 2}
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': 1, 'bar': 2}
    >>> x = Namespace({'foo': {'a': 1, 'b': 2}, 'bar': 3})
    >>> x.foo.a
    1
    >>> x.foo.b
    2
    >>> x.bar
    3
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': {'a': 1, 'b': 2}, 'bar': 3}
    >>> (lambda **kwargs: print(kwargs))(**x.foo)
    {'a': 1, 'b': 2}
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __delitem__(self, k):
        del self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, k):
        if not k.startswith('_'):
            if k not in self._data:
                return Namespace({})
            v = self._data[k]
            if isinstance(v, dict):
                v = Namespace(v)
            return v

        if k not in self.__dict__:
            raise AttributeError("'Namespace' object has no attribute '{}'".format(k))

        return self.__dict__[k]

    def __repr__(self):
        return repr(self._data)
