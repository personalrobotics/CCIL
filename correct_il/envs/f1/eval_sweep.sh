SEEDS='40 41 42 43 44 45 46 47 48 49'
QS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

for q in $QS; do
    for seed in $SEEDS; do
        python src/eval_all.py -pfx bc_seed${seed}_ -n 100 ../../../output/tune/f1/q${q}/seed${seed}/f110_gym:f110-v0/policy/* recordings/q${q} --map maps/random/map0
    done
done
