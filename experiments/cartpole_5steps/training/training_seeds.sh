
for seed in {0,5}
do
    for lambda in 0.0 0.3 0.5 0.7 0.9 0.99
    do 
        echo "Running script with seed: $seed and lambda: $lambda" 
        python 5step.py --seed "$seed" --lambda_val "$lambda" > 5GM${lambda}_seed${seed}.log 2>&1
    done
done



