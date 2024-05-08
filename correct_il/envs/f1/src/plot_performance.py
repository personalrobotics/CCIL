import argparse
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("out_path")
    return parser.parse_args()

def main():
    args = get_args()
    
    labels = []
    data = []

    for recording_path in args.recordings:
        with open(recording_path, "rb") as f:
            rec = pickle.load(f)
            rews = []
            for traj in rec["trajs"]:
                rews.append(np.sum(traj["rewards"]))
            label = os.path.basename(recording_path)[:-4]
            label = re.sub(r"seed\d+_", "", label)
            labels.append(label)
            data.append(rews)
    
    fig = plt.figure(figsize=(14, 4.8))
    ax = fig.add_subplot()
    ax.set_ylabel("Return")
    ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)

    fig.suptitle("Episode Return per Approach")
    fig.tight_layout()
    fig.savefig(args.out_path)
    fig.suptitle("BC Performance")

if __name__ == "__main__":
    main()