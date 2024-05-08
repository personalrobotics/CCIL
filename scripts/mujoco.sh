#!/bin/bash
SEEDS='40 41 42 43 44 45 46 47 48 49'

./scripts/train_ccil.sh "halfcheetah hopper walker2d" "${SEEDS}" "0.0001"
./scripts/train_ccil.sh "ant" "${SEEDS}" "1.0"
