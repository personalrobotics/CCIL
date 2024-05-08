#!/bin/bash
TASK='circle hover flythrugate'
SEED='41 42 43 44 45 46 47 48 49 50'
MODEL='spectral_normalization' # soft_sampling spectral_normalization slack none'
AUG='backward_euler' # backward_euler noisy_action forward_euler'
NOISE_BC="0.0001"

for seed in $SEED
    do
	for task in $TASK
        do
            # Train BC
            python correct_il/train_bc_policy.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed} policy.naive 1
            python correct_il/eval_bc_policy.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed} policy.naive 1

            # Train NoiseBC
            python correct_il/train_bc_policy.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed} policy.naive 1 policy.noise_bc ${NOISE_BC}
            python correct_il/eval_bc_policy.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed} policy.naive 1 policy.noise_bc ${NOISE_BC}

            # Use the config
            #python correct_il/train_dynamics_model.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed}
            #python correct_il/gen_aug_label.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed}
            #python correct_il/train_bc_policy.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed}
            #python correct_il/eval_bc_policy.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed}

            # Sweep
            for model in $MODEL
                do
                    python correct_il/train_dynamics_model.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed} dynamics.lipschitz_type ${model}
                    for aug in $AUG
                        do
                            python correct_il/gen_aug_label.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed} dynamics.lipschitz_type ${model} aug.type ${aug}
                            python correct_il/train_bc_policy.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed} dynamics.lipschitz_type ${model} aug.type ${aug}
                            python correct_il/eval_bc_policy.py config/${task}.yml output.location /gscratch/weirdlab/tycho/ccil_mujoco/${task} seed ${seed} dynamics.lipschitz_type ${model} aug.type ${aug}
                        done
                    done
            done
        done
