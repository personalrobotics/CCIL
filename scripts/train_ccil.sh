#!/bin/bash

if [[ $# -ne 3 ]]; then
    echo "Usage: ./scripts/train_ccil.sh 'TASKS...' 'SEEDS...' 'NOISE_BC'"
    exit 1
fi

TASKS="${1}"
SEEDS="${2}"
NOISE_BC="${3}"

for task in $TASKS; do
    for seed in $SEEDS; do
        printf "Train dynamics model\n"
        python correct_il/train_dynamics_model.py config/${task}.yml seed ${seed} output.location output/${task}

        printf "\n\nTrain naive BC!\n"
        python correct_il/train_bc_policy.py config/${task}.yml seed ${seed} policy.naive true output.location output/${task}

        printf "\n\nTrain noise BC! noise=${NOISE_BC}\n"
        python correct_il/train_bc_policy.py config/${task}.yml seed ${seed} policy.naive true policy.noise_bc ${NOISE_BC} output.location output/${task}

        printf "\n\nAug data\n"
        python correct_il/gen_aug_label.py config/${task}.yml seed ${seed} output.location output/${task}

        printf "\n\nTrain BC\n"
        python correct_il/train_bc_policy.py config/${task}.yml seed ${seed} output.location output/${task}

        if [ "${task}" != "f1" ]; then
            printf "\n\nEval\n"
            python correct_il/eval_bc_policy.py config/${task}.yml seed ${seed} policy.naive true output.location output/${task}
            python correct_il/eval_bc_policy.py config/${task}.yml seed ${seed} policy.naive true policy.noise_bc ${NOISE_BC} output.location output/${task}
            python correct_il/eval_bc_policy.py config/${task}.yml seed ${seed} output.location output/${task}
        fi
    done
    if [ "${task}" = "f1" ]; then
        printf "\n\nEval\n"
        pushd correct_il/envs/f1
        ./eval_all_seeds.sh
        popd
    fi
done

echo "Finished."
