#!/bin/bash
# Loop through seeds {0,5,10,15,20} and call the Python script with each seed

for seed in {0,5,10,15,20}
do        
    echo "Running script with seed: $seed "
    python3 DQN_2step_update2.py --seed "$seed" > DQN_2GM0_seed${seed}.log 2>&1
    
done



