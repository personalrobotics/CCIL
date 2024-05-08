import argparse
import yaml
import torch
import os
import gym
from gym.wrappers import TimeLimit
from tqdm import tqdm
import numpy as np
import pickle

from env_wrapper import F1EnvWrapper
from rewards import speed_reward
from run_expert import HORIZON
from state_featurizer import transform_state
from state_samplers import create_centerline_sampler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("save_path")
    parser.add_argument("-n", "--n_traj", type=int, default=1)
    parser.add_argument("--map", default="maps/circle")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

class D3Agent():
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    # For 1-batch query only!
    def predict(self, sample):
        with torch.no_grad():
            input = torch.from_numpy(sample).float().unsqueeze(0).to(self.device)
            at = self.policy(input)[0].cpu().numpy()
        return at

def main():
    args = get_args()

    # with open(args.config) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = "cpu" if args.cpu else "cuda"

    # policy = torch.jit.load(os.path.join(config["output"]["policy"], "policy.pt")).to(device)
    policy = torch.jit.load(os.path.join(args.config, "policy_final.pt")).to(device)
    agent = D3Agent(policy, device)

    env = gym.make("f110_gym:f110-v0", num_agents=1, map=args.map)
    env = F1EnvWrapper(env, create_centerline_sampler(), transform_state, speed_reward)
    env = TimeLimit(env, HORIZON)
    env: F1EnvWrapper

    trajs = []
    rews = []
    for _ in tqdm(range(args.n_traj)):
        done = False
        state = env.reset()
        traj = {"observations": [], "actions": [], "rewards": []}
        total_rew = 0
        while not done:
            obs = env.get_raw_state()
            action = agent.predict(state)
            state, rew, done, _ = env.step(action)
            total_rew += rew
            traj["observations"].append(obs)
            traj["actions"].append(action)
            traj["rewards"].append(rew)
        trajs.append(traj)
        rews.append(total_rew)
    print(f"Avg return: {np.mean(rews):.3f}, std={np.std(rews):.3f}")
    
    data = {
        "config": {
            "map": args.map,
            "config_path": args.config
        },
        "trajs": trajs
    }
    with open(args.save_path, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
