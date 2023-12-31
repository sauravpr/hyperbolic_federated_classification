#!/bin/bash

trails=10
seed=3
test_ratio=0.15
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
                                        --test_ratio=$test_ratio \
                                        --ref_k=3 \
                                        --CE --CP --FLE --FLP --multiclass

            done
            
        done
        
    done

done
