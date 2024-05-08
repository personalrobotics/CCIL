from tabulate import tabulate
import re
import numpy as np
import argparse
import pickle
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_files", nargs="+")
    return parser.parse_args()

def get_rew_files(path):
    parent_dir = os.path.dirname(os.path.abspath(path))
    files = os.listdir(parent_dir)
    rewards_files = [f for f in files if "rewards_noise" in f]
    rewards_files.sort(key=lambda f: float(f[len("rewards_noise_"):-4]))
    return [os.path.join(parent_dir, p) for p in rewards_files]

def read_results(results_file):
    results_files = get_rew_files(results_file)
    noise_results = []
    for results_file in results_files:
        with open(results_file, "rb") as f:
            data = pickle.load(f)
        path = os.path.normpath(results_file).split(os.sep)
        policy_name = path[-2]
        task = path[-4]
        print(f"{results_file} {np.mean(data)}")
        noise_results.append(np.mean(data))
    return f"{task}_{policy_name}", noise_results[0]

def main():
    args = get_args()
    env_scores = {}
    for results_file in args.results_files:
        env, score = read_results(results_file)
        if env not in env_scores:
            env_scores[env] = []
        env_scores[env].append(score)
    
    table = [["Task", "Score", "Score (std)", "# seeds"]]
    for task in env_scores:
        print(task)
        score = np.mean(env_scores[task])
        score_std = np.std(env_scores[task])
        table.append([task, score, score_std, len(env_scores[task])])
    print(tabulate(table, headers="firstrow", floatfmt=".3f", tablefmt="grid"))


if __name__ == "__main__":
    main()
