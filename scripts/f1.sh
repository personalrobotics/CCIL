#!/bin/bash
TASKS='f1'
SEEDS='40 41 42 43 44 45 46 47 48 49'
NOISE_BC='0.01'

./scripts/train_ccil.sh "${TASKS}" "${SEEDS}" "${NOISE_BC}"
