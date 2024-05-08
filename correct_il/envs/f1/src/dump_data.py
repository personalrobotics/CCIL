import argparse
import pickle
import numpy as np
from tqdm import tqdm

from state_featurizer import transform_state_
from env_wrapper import get_centerline

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_path")
    parser.add_argument("out_path")
    return parser.parse_args()

def main():
    args = get_args()
    with open(args.recording_path, "rb") as f:
        recording = pickle.load(f)
    
    centerline = get_centerline(recording["config"]["map"])
    
    dataset = []
    for traj in tqdm(recording["trajs"]):
        observations = []
        for o_raw in traj["observations"]:
            o = transform_state_(centerline, o_raw)
            observations.append(o)
        actions = np.array(traj["actions"])
        rewards = np.array(traj["rewards"])
        dataset.append({
            "observations": np.array(observations),
            "actions": actions,
            "rewards": rewards
        })

    with open(args.out_path, "wb") as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    main()