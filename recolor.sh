#!/bin/bash
python recolor.py --output_path predictions/DUNet16_0.5_color/ --model_type DUNet16 --problem_type instruments --model_path runs/debug/ --fold 0 --batch-size 2 --jaccard-weight 0.1








