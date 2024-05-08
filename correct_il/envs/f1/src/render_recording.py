from f110_gym.envs.base_classes import Integrator
import yaml
import argparse
import numpy as np
from env_wrapper import F1EnvWrapper
import gym
import cv2
import pygame
from pygame import surfarray
from tqdm import tqdm
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_path", help="Path to recording file")
    parser.add_argument("vid_path", help="Path to save video")
    parser.add_argument("--fps", default=20, type=int)
    parser.add_argument("-rm", "--render_mode", default="human", help="Render mode (human or human_zoom)")
    parser.add_argument("-i", "--index", default=0, type=int, help="Index of trajectory in recording.")
    return parser.parse_args()

def pg_to_cv2(cvarray:np.ndarray)->np.ndarray:
    cvarray = cvarray.swapaxes(0, 1) #rotate
    cvarray = cv2.cvtColor(cvarray, cv2.COLOR_RGB2BGR) #RGB to BGR
    return cvarray

def save_frame(writer: cv2.VideoWriter, screen: pygame.Surface):
    pg_frame = surfarray.array3d(screen)
    cv_frame = pg_to_cv2(pg_frame)
    writer.write(cv_frame)

def get_poses(states):
    poses = np.empty((len(states), 3))
    for i, state in enumerate(states):
        ego_idx = state["ego_idx"]
        x = state["poses_x"][ego_idx]
        y = state["poses_y"][ego_idx]
        theta = state["poses_theta"][ego_idx]
        poses[i] = [x, y, theta]
    return poses


def main():
    args = get_args()
    with open(args.recording_path, "rb") as f:
        data = pickle.load(f)
    states = data["trajs"][args.index]["observations"]
    poses = get_poses(states)
    map = data["config"]["map"]

    env = gym.make("f110_gym:f110-v0", integrator=Integrator.Euler, num_agents=1, map=map)
    env = F1EnvWrapper(env, lambda *_, **__: np.zeros((1,3)), lambda *_, **__: np.zeros(1), lambda *_, **__: 0.0)
    
    control_hz = round(1 / (env.timestep * env.action_repeat))

    # adjust thetas to be +-pi radians from the prev theta
    for i in range(1, len(poses)):
        prev_theta = poses[i-1,2]
        theta = poses[i, 2]
        poses[i, 2] = prev_theta + np.arctan2(np.sin(theta-prev_theta), np.cos(theta-prev_theta))

    points = poses[:,:2]
    print("Traveled dist:", np.sum(np.linalg.norm(points[1:] - points[:-1], axis=-1)))
    print(f"Lap time (assuming 20hz control): {len(points) / control_hz:.3f}sec")

    vid_writer = None

    timestamps = np.arange(0, poses.shape[0] / control_hz, 1/control_hz)

    env.reset()
    for vid_ts in tqdm(np.arange(0, poses.shape[0] / control_hz, 1/args.fps)):
        i = int(vid_ts * control_hz)
        if i == poses.shape[0]-1:
            dist = np.linalg.norm(poses[i-1,:2] - poses[i,:2])
        else:
            dist = np.linalg.norm(poses[i,:2] - poses[i+1,:2])
        vel = dist * control_hz
        pose_x = np.interp(vid_ts, timestamps, poses[:,0])
        pose_y = np.interp(vid_ts, timestamps, poses[:,1])
        pose_theta = np.interp(vid_ts, timestamps, poses[:,2])
        env.curr_state["poses_x"][0] = pose_x
        env.curr_state["poses_y"][0] = pose_y
        env.curr_state["poses_theta"][0] = pose_theta
        env.curr_state["linear_vels_x"][0] = vel
        env.render(mode=args.render_mode)
        if vid_writer is None:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            vid_writer = cv2.VideoWriter(args.vid_path, codec, args.fps, env.screen.get_size())
        if vid_writer:
            save_frame(vid_writer, env.screen)
    if vid_writer:
        vid_writer.release()

if __name__ == "__main__":
    main()