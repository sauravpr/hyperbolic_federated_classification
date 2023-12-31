#!/bin/bash
trails=5
test_ratio=0.10
C=20000
max_iter_SVM=200000
ref_k=1
R=0.95
for num_clients in 10
do

    for a_r in 1 2 3 4  
    do

        for gamma in 1
        do

            for eps in 2.0 4.0 6.0 8.0 10.0 15.0 20.0
            do
                
                for part_mode in "iid"
                do
                    
                    for dataset_name in "uniform_H"
                    do
                        python working_version_uniform.py --dataset_name=$dataset_name \
                                                --num_clients=$num_clients \
                                                --a_r=$a_r \
                                                --ref_k=$ref_k \
                                                --C=$C \
                                                --max_iter_SVM=$max_iter_SVM \
                                                --gamma=$gamma \
                                                --switch_ratio=0.0 \
                                                --eps=$eps \
                                                --test_ratio=$test_ratio \
                                                --R=$R \
                                                --trails=$trails \
                                                --part_mode=$part_mode
                    done
                    
                done
                
            done
        done

    done
done