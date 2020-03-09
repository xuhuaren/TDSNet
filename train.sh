#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 1 --fold 0 --workers 4 --lr 0.0001 --n-epochs 50 --model TDSNet --alpha 0.2 --beta 0.0 --gama 0.0
wait
CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 1 --fold 0 --workers 4 --lr 0.00005 --n-epochs 25 --model TDSNet --alpha 0.2 --beta 0.4 --gama 0.0
wait
CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 1 --fold 0 --workers 4 --lr 0.00003 --n-epochs 25 --model TDSNet --alpha 0.2 --beta 0.4 --gama 0.2



