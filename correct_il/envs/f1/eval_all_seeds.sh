SEEDS='40 41 42 43 44 45 46 47 48 49'

for seed in $SEEDS; do
    python src/eval_all.py -pfx bc_seed${seed}_ -n 100 ../../../output/f1/seed${seed}/f110_gym:f110-v0/policy/* recordings --map maps/random/map0
done
