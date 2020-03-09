#!/bin/bash
conda install pytorch=0.4.1 cuda90 -c pytorch
wait
conda install -c pytorch torchvision
wait
pip install -r util/requirement.txt