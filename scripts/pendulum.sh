#!/bin/bash
TASK='pendulum_disc pendulum_cont'
SEED='41 42 43 44 45 46 47 48 49 50'
NOISE_BC='0.0001'

./scripts/train_ccil.sh "${TASK}" "${SEED}" "${NOISE_BC}"

# Add ablation to train dynamics model without enforcing Lipschitz continuity
MODEL='none'
for task in $TASK
    do
    for seed in $SEED
        do
            for model in $MODEL
                do
                    python correct_il/train_dynamics_model.py config/${task}.yml output.location output/${task} seed ${seed} dynamics.lipschitz_type ${model}
                    python correct_il/gen_aug_label.py config/${task}.yml output.location output/${task} seed ${seed} dynamics.lipschitz_type ${model}
                    python correct_il/train_bc_policy.py config/${task}.yml output.location output/${task} seed ${seed} dynamics.lipschitz_type ${model}
                    python correct_il/eval_bc_policy.py config/${task}.yml output.location output/${task} seed ${seed} dynamics.lipschitz_type ${model}
            done
        done
    done
