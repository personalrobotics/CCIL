import os
import pickle
import torch
import numpy as np

from tqdm import tqdm

from models.nn_dynamics import WorldModel
from utils import parse_config
from gen_aug_label import model_fits_data, choose_augmentation, construct_parser


def main():
    parser = construct_parser()
    config = parse_config(parser)
    output_folder = config.output.aug
    output_path = os.path.join(output_folder, "aug_data.pkl")
    os.makedirs(output_folder, exist_ok=True)
    print(config)

    if os.path.isfile(output_path) and not config.overwrite:
        print(f"Found existing results in {output_folder}, quit")
        exit(0)

    model_path = os.path.join(config.output.dynamics, "dynamics.pkl")
    with open(model_path, "rb") as f:
        model: WorldModel = pickle.load(f)

    with open(config.data.pkl, "rb") as f:
        data = pickle.load(f)

    if not config.aug.chunk_size:
        print("WARNING: Chunk size not specified, using default value 8")
        chunk_size = config.aug.chunk_size
    else:
        chunk_size = 8

    aug_fn = choose_augmentation(config.aug)

    gen_data = []
    n_succ = n_fail = 0
    for traj in tqdm(data, desc="Generating new trajectories"):
        states = torch.from_numpy(traj["observations"]).float().cuda()
        actions = torch.from_numpy(traj["actions"]).float().cuda()
        for i in tqdm(range(0, len(states)-1), desc="Generating chunks", leave=False):
            s, a, sp = states[i], actions[i], states[i+1]
            if model_fits_data(model, s, a, sp, config.aug.model_err_thresh):
                _succ, _fail, (gen_s, gen_a, gen_sp), info = aug_fn(model, (s, a, sp))
                assert _succ
                if _succ:
                    gen_a_chunk = np.concatenate([gen_a[None, :], actions[i:i+chunk_size-1].cpu().numpy()])
                    if len(gen_a_chunk) < chunk_size:
                        gen_a_chunk = np.concatenate([
                            gen_a_chunk,
                            np.tile(gen_a_chunk[-1:], (chunk_size - len(gen_a_chunk), 1))
                        ])
                    assert len(gen_a_chunk) == chunk_size
                    gen_s_dummy = np.zeros((chunk_size, gen_s.shape[-1]), dtype=gen_s.dtype)
                    gen_s_dummy[0] = gen_s
                    gen_data.append({"observations": gen_s_dummy, "actions": gen_a_chunk})
            else:
                _succ, _fail, gen_data, info = 0, 1, None, None
            n_succ += _succ
            n_fail += _fail

    all_data = data + gen_data

    print(f"Generated {n_succ} successful and {n_fail} failed chunks")
    assert n_succ == len(gen_data)

    with open(output_path, "wb") as f:
        pickle.dump(gen_data, f)

    with open(os.path.join(output_folder, "all_data.pkl"), "wb") as f:
        pickle.dump(all_data, f)

if __name__ == "__main__":
    main()
