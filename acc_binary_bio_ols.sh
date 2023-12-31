#!/bin/bash

trails=10
seed=10
for num_clients in 3 
do
    
    for eps in 0.001 0.01 0.1
    do
        
        for part_mode in "iid"
        do
            
            for dataset_name in "olsson"
            do
                python working_version.py --dataset_name=$dataset_name \
                                        --num_clients=$num_clients \
                                        --switch_ratio=0.0 \
                                        --eps=$eps \
                                        --part_mode=$part_mode \
                                        --seed=$seed \
                                        --trails=$trails \
                                        --test_ratio=0.3 \
                                        --ref_k=3 \
                                        --CE --CP --FLE --FLP

            done
            
        done
        
    done

done
