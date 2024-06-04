#!/bin/bash

TASK="$@"
SEED='40 41 42 43 44 45 46 47 48 49'
DATA_FRAC="0.2 0.4 0.6 0.8 1.0"
OUT_ROOT="output/data_ablation"

for task in $TASK; do
    for p in $DATA_FRAC; do
        OUT_DIR="${OUT_ROOT}/${task}/data_frac${p}"
        for seed in $SEED; do
            # CCIL
            # printf "Train dynamics model\n"
            # python correct_il/train_dynamics_model.py config/${task}.yml seed ${seed} data.data_frac ${p} output.location ${OUT_DIR}
            # printf "\n\nAug data\n"
            # python correct_il/gen_aug_label.py config/${task}.yml seed ${seed} data.data_frac ${p} output.location ${OUT_DIR}
            # printf "\n\nTrain BC\n"
            # python correct_il/train_bc_policy.py config/${task}.yml seed ${seed} data.data_frac ${p} output.location ${OUT_DIR}
            # printf "\n\nEval\n"
            # python correct_il/eval_bc_policy.py config/${task}.yml seed ${seed} data.data_frac ${p} output.location ${OUT_DIR}

            # BC
            python correct_il/train_bc_policy.py config/${task}.yml seed ${seed} policy.naive true data.data_frac ${p} output.location ${OUT_DIR}
            python correct_il/eval_bc_policy.py config/${task}.yml seed ${seed} policy.naive true data.data_frac ${p} output.location ${OUT_DIR}
        done
    done
done