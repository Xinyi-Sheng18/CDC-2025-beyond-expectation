

for seed in {0,5}
do
    for lambda in 0.0 0.3 0.5 0.7 0.9 0.99
    do
        echo "Testing seed: $seed and lambda: $lambda"
        python3 Testing.py --seed $seed --lambda_val $lambda > test_log_${lambda}_seed${seed}.log 2>&1
    done
done

#!/bin/bash
# Loop through seeds 1 to 10 and call the Python script with each seed
'''
for seed in  
do
    echo "Running script with seed: $seed"
    python3 Testing_GM.py --seed "$seed" > Testing_GM_seed${seed}.log 2>&1
done
'''


