#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python test.py --batch-size 1 --fold 0 --workers 4 --model_type TDSNet --model_path ./runs/debug



