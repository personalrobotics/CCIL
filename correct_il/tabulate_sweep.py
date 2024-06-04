from tabulate import tabulate
import numpy as np
import argparse
import pickle
import os
import glob
from collections import defaultdict

TABLE_HEADER = ["Task", "Policy", "Swept Param", "Score", "Score (std)", "# seeds"]
TASK_GROUPS = {
    "mujoco": {"ant", "halfcheetah", "hopper", "walker2d"},
    "metaworld": {"button", "coffee_pull", "coffee_push", "drawer_close"},
    "drone": {"hover", "circle", "flythrugate"},
    "pendulum": {"pendulum_cont", "pendulum_disc"}
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", help="Root directory of sweep")
    parser.add_argument("-t", "--task", help="Leave unspecified to show best from all tasks")
    return parser.parse_args()

def read_single_results(results_file):
    with open(results_file, "rb") as f:
        data = pickle.load(f)
    path = os.path.normpath(results_file).split(os.sep)
    policy_name = path[-2]
    # print(f"{results_file} {np.mean(data)}")
    return policy_name, np.mean(data)

def get_task_results(task_dir):
    scores = defaultdict(lambda: defaultdict(list))
    for hparam in os.listdir(task_dir):
        for path in glob.glob(os.path.join(task_dir, hparam, "**", "rewards*.pkl"), recursive=True):
            policy, score = read_single_results(path)
            scores[policy][hparam].append(score)
    return scores

def best_policy(scores_dict: dict):
    best = None
    for policy in scores_dict:
        hparam = max((hparam for hparam in scores_dict[policy]), key=lambda h: np.mean(scores_dict[policy][h]))
        if best is None or np.mean(scores_dict[policy][hparam]) > np.mean(scores_dict[best[0]][best[1]]):
            best = (policy, hparam)
    return best

def show_task_sweep(task, task_dir):
    scores = get_task_results(task_dir)
    table = []
    for policy in sorted(scores):
        for hparam in sorted(scores[policy]):
            score_arr = scores[policy][hparam]
            score = np.mean(score_arr)
            score_std = np.std(score_arr)
            table.append([task, policy, hparam, score, score_std, len(score_arr)])
    print(tabulate(table, TABLE_HEADER, tablefmt="grid", floatfmt=".3f"))

def show_best_all_tasks(sweep_dir):
    task_scores = {}
    for task in os.listdir(sweep_dir):
        scores = get_task_results(os.path.join(sweep_dir, task))
        task_scores[task] = scores

    def get_row(task):
        policy, hparam = best_policy(task_scores[task])
        score_arr = task_scores[task][policy][hparam]
        score = np.mean(score_arr)
        score_std = np.std(score_arr)
        return [task, policy, hparam, score, score_std, len(score_arr)]

    for task_group in sorted(TASK_GROUPS):
        table = [get_row(task) for task in sorted(TASK_GROUPS[task_group]) if task in task_scores]
        print(f"Task Suite: {task_group}")
        print(tabulate(table, TABLE_HEADER, tablefmt="grid", floatfmt=".3f"))
        print()

    ALL_TASKS = set.union(*TASK_GROUPS.values())
    table = [get_row(task) for task in sorted(task_scores) if task not in ALL_TASKS]
    if len(table) > 0:
        print(f"Task Suite: Miscellaneous")
        print(tabulate(table, headers=TABLE_HEADER, floatfmt=".3f", tablefmt="grid"))

def main():
    args = get_args()

    if args.task:
        show_task_sweep(args.task, os.path.join(args.sweep_dir, args.task))
    else:
        show_best_all_tasks(args.sweep_dir)


if __name__ == "__main__":
    main()
