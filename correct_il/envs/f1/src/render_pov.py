import argparse
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pygame
import yaml

from f110_gym.envs.base_classes import Integrator
from env_wrapper import F1EnvWrapper
from rewards import speed_reward
from state_featurizer import transform_state
from state_samplers import create_centerline_sampler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("map")
    parser.add_argument("save_path")
    return parser.parse_args()

def pos_m_to_px(map_cfg, screen: pygame.Surface, pos):
    origin = np.array(map_cfg["origin"][:2])
    m_per_px = map_cfg["resolution"]
    pos_px = np.around((pos - origin) / m_per_px).astype(int)
    pos_px[1] = screen.get_height() - pos_px[1]
    return pos_px

def main():
    args = get_args()
    env = gym.make("f110_gym:f110-v0", integrator=Integrator.Euler, num_agents=1, map=args.map)
    env = F1EnvWrapper(env, create_centerline_sampler(), transform_state, speed_reward)

    with open(env.map_name + ".yaml", "r") as f:
        map_cfg = yaml.load(f, Loader=yaml.FullLoader)

    env.reset()
    obs = env.get_raw_state()
    print(obs)
    scan = np.array(obs["scans"][0])
    angles = np.linspace(-4.7/2, 4.7/2, scan.shape[0])

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(scan.reshape(1,-1), cmap="jet_r")
    ax.set_position([0, 0, 1, 1])
    ax.axis("tight")
    fig.savefig(args.save_path)

    env.render()
    angles += obs["poses_theta"][0]

    pc_x = np.cos(angles) * scan + obs["poses_x"][0]
    pc_y = np.sin(angles) * scan + obs["poses_y"][0]
    max_dist = np.max(scan)
    min_dist = np.min(scan)
    scan_normalized = (scan - min_dist) / (max_dist - min_dist)
    cmap = matplotlib.colormaps["jet_r"]
    for (d, x,y) in zip(scan_normalized, pc_x, pc_y):
        color = cmap(d, bytes=True)[:3]
        pos = pos_m_to_px(map_cfg, env.screen, (x, y))
        pygame.draw.circle(env.screen, color, pos, radius=2)

    pygame.image.save(env.screen, f"{args.save_path[:-4]}_map.png")

if __name__ == "__main__":
    main()