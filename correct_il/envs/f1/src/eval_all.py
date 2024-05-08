import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_dirs", nargs="+")
    parser.add_argument("out_dir")
    parser.add_argument("-n", "--n_traj", type=int, default=100)
    parser.add_argument("-pfx", "--prefix", default="bc_")
    return parser.parse_known_args()

def clean_name(name: str):
    name = name.replace("backward_euler", "be")
    name = name.replace("spectral_normalizationL", "spectral_")
    return name

def main():
    args, sub_args = get_args()
    for policy_dir in args.policy_dirs:
        print(f"Evaluating {policy_dir}")
        name = os.path.basename(policy_dir)
        name = clean_name(name)
        config_path = os.path.join(policy_dir, "config.yml")
        save_path = os.path.join(args.out_dir, f"{args.prefix}{name}.pkl")
        os.system(f"python src/eval_bc.py -n {args.n_traj} {config_path} {save_path} {' '.join(sub_args)}")

if __name__ == "__main__":
    main()