import numpy as np 

accum_rew_list = []

for seed in {0,5,10,15,20}:
    filename = f'Testing_result_2GM0_seed{seed}.npy'
    rew = np.load( filename )
    accum_rew_list.append(rew)

print( f"Median of acuumulative rewards from different seeds:{ np.median(accum_rew_list) }" )
print( f"Mean of acuumulative rewards from different seeds:{ np.mean(accum_rew_list) }" )

# python Combination.py > Combination.log 2>&1
