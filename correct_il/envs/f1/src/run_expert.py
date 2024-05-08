import argparse
import gym
from gym.wrappers import TimeLimit
import pickle
from tqdm import tqdm
import numpy as np

from env_wrapper import F1EnvWrapper
from rewards import speed_reward
from state_featurizer import transform_state
from state_samplers import create_constant_sampler, create_centerline_sampler
from pure_pursuit import PurePursuit

CONTROL_HZ = 20
HORIZON_SEC = 40
HORIZON = HORIZON_SEC * CONTROL_HZ

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", help="Where to save recording file to")
    parser.add_argument("-m", "--map", default="maps/circle")
    parser.add_argument("-n", "--n_traj", type=int, default=1)
    parser.add_argument("-l", "--lookahead", type=float)
    parser.add_argument("-kp", "--turn_kp", type=float)
    parser.add_argument("-v", "--vel", type=float)
    parser.add_argument("-r", "--render", action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    env = gym.make("f110_gym:f110-v0", num_agents=1, map=args.map)
    env = F1EnvWrapper(env, create_centerline_sampler(), transform_state, speed_reward)
    env = TimeLimit(env, HORIZON)
    env: F1EnvWrapper

    pp_kwargs = {}
    if args.lookahead:
        pp_kwargs["lookahead"] = args.lookahead
    if args.turn_kp:
        pp_kwargs["turn_kp"] = args.turn_kp
    if args.vel:
        pp_kwargs["vel"] = args.vel
    expert = PurePursuit(env.centerline, **pp_kwargs)
    
    trajs = []
    rews = []
    for _ in tqdm(range(args.n_traj)):
        done = False
        env.reset()
        traj = {"observations": [], "actions": [], "rewards": []}
        total_rew = 0
        while not done:
            state = env.get_raw_state()
            action = expert.get_action(state)
            _, rew, done, _ = env.step(action)
            total_rew += rew
            traj["observations"].append(state)
            traj["actions"].append(action)
            traj["rewards"].append(rew)
            if args.render:
                env.render()
        if len(traj["observations"]) != HORIZON:
            print(f"WARN: Expert terminated early! Traj len: {len(traj['observations'])}")
        trajs.append(traj)
        rews.append(total_rew)
    print(f"Avg return: {np.mean(rews):.3f}, std={np.std(rews):.3f}")
    
    data = {
        "config": {
            "map": args.map,
            "pure_pursuit": {
                "lookahead": expert.lookahead,
                "turn_kp": expert.turn_kp,
                "vel": expert.vel
            }
        },
        "trajs": trajs
    }
    with open(args.save_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()