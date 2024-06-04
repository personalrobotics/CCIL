#!/bin/bash

TASK="$@"
SEED='40 41 42 43 44 45 46 47 48 49'
QUANTILE="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
OUT_ROOT="output/tune"

for task in $TASK; do
    for q in $QUANTILE; do
        OUT_DIR="${OUT_ROOT}/${task}/q${q}"
        for seed in $SEED; do
            printf "Train dynamics model\n"
            python correct_il/train_dynamics_model.py config/${task}.yml seed ${seed} aug.label_err_quantile ${q} output.location ${OUT_DIR}
            printf "\n\nAug data\n"
            python correct_il/gen_aug_label.py config/${task}.yml seed ${seed} aug.label_err_quantile ${q} output.location ${OUT_DIR}
            printf "\n\nTrain BC\n"
            python correct_il/train_bc_policy.py config/${task}.yml seed ${seed} aug.label_err_quantile ${q} output.location ${OUT_DIR}
            # printf "\n\nEval\n"
            # python correct_il/eval_bc_policy.py config/${task}.yml seed ${seed} aug.label_err_quantile ${q} output.location ${OUT_DIR}
        done
    done
done
