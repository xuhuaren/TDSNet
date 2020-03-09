#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python predict_evaluation.py --batch-size 1 --fold 0 --workers 4 --model DUNet16 --model_path ./runs/debug/model_0_DUNet16_0.1_only_class.pt



