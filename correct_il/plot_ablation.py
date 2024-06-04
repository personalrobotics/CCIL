import numpy as np
import argparse
import pickle
import os
import glob
from collections import defaultdict
from matplotlib import pyplot as plt
import re

# Configure fonts
SMALLEST_SIZE = 7
SMALL_SIZE = 7
MEDIUM_SIZE = 7
BIGGER_SIZE = 7
plt.rc('font', size=SMALLEST_SIZE, family="Times New Roman")
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALLEST_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE, figsize=(7/4,21/16))

task_cfg = {
    "circle" : "Drone: Circle",
    "halfcheetah" : "Mujoco: HalfCheetah",
    "button" : "Metaworld: ButtonPress",
}

plot_cfg = {
    "ccil": {
        "name": "CCIL",
        "marker": "o",
        "color": "#d00",
        "bar_color": "#c00",
        "shift": 0.0
    },
    "naive": {
        "name": "BC",
        "marker": "o",
        "color": "#b3cfff",
        "bar_color": "#b3cfff",
        "shift": -0.4
    }
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ablation_dir", help="Root directory of ablation")
    parser.add_argument("-o", "--out_dir", help="Where to save plots. If unspecified, will only display.")
    return parser.parse_args()

def read_single_results(results_file):
    with open(results_file, "rb") as f:
        data = pickle.load(f)
    path = os.path.normpath(results_file).split(os.sep)
    policy_name = path[-2]
    return policy_name, np.mean(data)

def get_task_results(task_dir):
    scores = defaultdict(lambda: defaultdict(list))
    for hparam in os.listdir(task_dir):
        for path in glob.glob(os.path.join(task_dir, hparam, "**", "rewards*.pkl"), recursive=True):
            policy, score = read_single_results(path)
            scores[policy][hparam].append(score)
    return scores

def main():
    args = get_args()
    task_scores = {}
    for task in os.listdir(args.ablation_dir):
        scores = get_task_results(os.path.join(args.ablation_dir, task))
        task_scores[task] = scores

    for task in task_scores:
        fig = plt.figure()
        ax = fig.add_subplot()
        all_x = []
        all_y = []
        for policy in sorted(task_scores[task]):
            hparam_pairs = [(float(re.sub(r"[^\d.]", "", hparam)), hparam) for hparam in task_scores[task][policy]]
            hparam_pairs.sort()
            x = [p[0] for p in hparam_pairs]
            y = [np.mean(task_scores[task][policy][p[1]]) for p in hparam_pairs]
            all_x.extend(x)
            all_y.extend(y)

            cfg = plot_cfg.get(policy, plot_cfg["ccil"])

            ax.plot(x, y, label=cfg["name"], marker=cfg["marker"], color=cfg["color"], clip_on=False, markersize=2, markeredgecolor=cfg["color"], markerfacecolor="#fff", markeredgewidth=2, linewidth=2, zorder=3)
        ax.legend()
        ax.set_title(task_cfg.get(task, task))
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xticks([min(all_x), max(all_x)])
        ax.set_yticks([min(all_y), max(all_y)])
        ax.ticklabel_format(style="sci", scilimits=(0,0))
        ax.set_xlabel("Proportion of Data", labelpad=-MEDIUM_SIZE)
        ax.set_ylabel("Return", labelpad=-MEDIUM_SIZE)
        fig.tight_layout(pad=0)
        fig.show()
        if args.out_dir:
            os.makedirs(os.path.join(args.out_dir, "png"), exist_ok=True)
            os.makedirs(os.path.join(args.out_dir, "pdf"), exist_ok=True)
            fig.savefig(os.path.join(args.out_dir, f"png/{task}_ablation_line_plot.png"), transparent=True, pad_inches=0, bbox_inches="tight", dpi=300)
            fig.savefig(os.path.join(args.out_dir, f"pdf/{task}_ablation_line_plot.pdf"), format="pdf", transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
