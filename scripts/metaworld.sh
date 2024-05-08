#!/bin/bash
TASKS='coffee_pull button coffee_push drawer_close'
SEEDS='40 41 42 43 44 45 46 47 48 49'
NOISE_BC='0.0001'

./scripts/train_ccil.sh "${TASKS}" "${SEEDS}" "${NOISE_BC}"
