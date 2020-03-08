#!/bin/bash




for i in 0.1
do
    CUDA_VISIBLE_DEVICES=7 python train.py --device-ids 0 --batch-size 1 --fold 0 --workers 4 --lr 0.0001 --n-epochs 80 --jaccard-weight 0.5 --model DUNet16 --type instruments --loss_weight $i
done


