from f110_gym.envs.base_classes import Integrator
import yaml
import argparse
import numpy as np
from env_wrapper import F1EnvWrapper
import gym
import pygame
from tqdm import tqdm
import pickle
import colorsys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_path", help="Path to recording file")
    parser.add_argument("img_path", help="Path to save image")
    parser.add_argument("-n", "--n_traj", default=-1, type=int, help="Number of trajectories to render. -1 to render all.")
    parser.add_argument("-rp", "--render_period", type=float, default=0.2)
    return parser.parse_args()

def get_poses(states):
    poses = np.empty((len(states), 3))
    for i, state in enumerate(states):
        ego_idx = state["ego_idx"]
        x = state["poses_x"][ego_idx]
        y = state["poses_y"][ego_idx]
        theta = state["poses_theta"][ego_idx]
        poses[i] = [x, y, theta]
    return poses

def pos_m_to_px(map_cfg, screen: pygame.Surface, pos):
    origin = np.array(map_cfg["origin"][:2])
    m_per_px = map_cfg["resolution"]
    pos_px = np.around((pos - origin) / m_per_px).astype(int)
    pos_px[1] = screen.get_height() - pos_px[1]
    return pos_px

def draw_car(map_cfg, params, screen, pose, color):
    m_per_px = map_cfg["resolution"]
    pos_px = pos_m_to_px(map_cfg, screen, pose[:-1])
    car_len_px = params["length"] / m_per_px
    car_width_px = params["width"] / m_per_px
    theta = pose[-1]
    rotmat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    cob = np.array([[1., 0.], [0., -1.]]) # change of basis required to invert y axis
    car_poly = [
        pos_px + cob @ rotmat @ np.array([car_len_px/2, 0]),
        pos_px + cob @ rotmat @ np.array([-car_len_px/2, -car_width_px/2]),
        pos_px + cob @ rotmat @ np.array([-car_len_px/2, car_width_px/2])
    ]
    car_poly = [np.around(x).astype(int) for x in car_poly]
    pygame.draw.polygon(screen, color, car_poly)

def draw_dashed_line(screen: pygame.Surface, color, start, end, length, gap, width=1):
    start = np.array(start)
    end = np.array(end)
    direction = (end - start).astype(np.float32)
    direction /= np.linalg.norm(direction)

    draw_start = start
    draw_end = start + direction * length
    # import pdb ; pdb.set_trace()
    while np.dot(start - draw_end, end - draw_end) < 0:
        pygame.draw.line(screen, color, draw_start.round().astype(int), draw_end.round().astype(int), width=width)
        draw_start = draw_end + direction * gap
        draw_end = draw_start + direction * length
    if np.dot(start - draw_start, end - draw_start) < 0:
        pygame.draw.line(screen, color, draw_start.round().astype(int), end, width=width)


def main():
    pygame.init()
    args = get_args()
    with open(args.recording_path, "rb") as f:
        data = pickle.load(f)
        print(data["config"])
    
    map = "maps/random/map0" #data["config"]["map"]
    env = gym.make("f110_gym:f110-v0", integrator=Integrator.Euler, num_agents=1, map=map)
    env = F1EnvWrapper(env, lambda *_, **__: np.zeros((1,3)), lambda *_, **__: np.zeros(1), lambda *_, **__: 0.0)

    map_img = pygame.image.load(env.map_name + env.map_ext)
    with open(env.map_name + ".yaml", "r") as f:
        map_cfg = yaml.load(f, Loader=yaml.FullLoader)
    screen = pygame.display.set_mode((map_img.get_width(), map_img.get_height()))
    screen.blit(map_img, (0, 0))

    n_traj = args.n_traj if args.n_traj > 0 else len(data["trajs"])
    for traj_idx in tqdm(range(n_traj)):
        states = data["trajs"][traj_idx]["observations"]
        poses = get_poses(states)
        control_hz = round(1 / (env.timestep * env.action_repeat))

        # adjust thetas to be +-pi radians from the prev theta
        for i in range(1, len(poses)):
            prev_theta = poses[i-1,2]
            theta = poses[i, 2]
            poses[i, 2] = prev_theta + np.arctan2(np.sin(theta-prev_theta), np.cos(theta-prev_theta))

        traj_dur = poses.shape[0] / control_hz
        timestamps = np.arange(0, traj_dur, 1/control_hz)
        color = [int(round(x*255)) for x in colorsys.hsv_to_rgb(traj_idx / n_traj, 1, 1)]

        env.reset()
        draw_car(map_cfg, env.params, screen, poses[0], color)
        last_point = pos_m_to_px(map_cfg, screen, poses[0, :2])
        for ts in np.arange(args.render_period, traj_dur/5, args.render_period):
            # import pdb ; pdb.set_trace()
            i = int(ts * control_hz)
            pose_x = np.interp(ts, timestamps, poses[:,0])
            pose_y = np.interp(ts, timestamps, poses[:,1])
            pos_px = pos_m_to_px(map_cfg, screen, [pose_x, pose_y])
            # pygame.draw.line(screen, color, last_point, pos_px, width=2)
            draw_dashed_line(screen, color, last_point, pos_px, 0.5 / map_cfg["resolution"], 0.5 / map_cfg["resolution"], width=1)
            last_point = pos_px
            # draw_car(map_cfg, env.params, screen, [pose_x, pose_y, pose_theta], color)
    pygame.image.save(screen, args.img_path)

if __name__ == "__main__":
    main()