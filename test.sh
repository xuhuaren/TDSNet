#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py --batch-size 1 --fold 0 --workers 4 --model TDSNet --model_path ./runs/debug/model_0_TDSNet.pt



