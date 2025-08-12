#!/bin/bash
# Loop through seeds {0,5,10,15,20} and call the Python script with each seed for each lambda

for seed in {0,5,10,15,20}
do
    for lambda in 0.0 0.3 0.5 0.7 0.9 0.99
    do 
        echo "Running script with seed: $seed and lambda: $lambda"
        python3 DQN_2step_update2.py --seed "$seed" --lambda_val "$lambda" > DQN_2GM_seed${seed}_lambda${lambda}.log 2>&1
    done
done



