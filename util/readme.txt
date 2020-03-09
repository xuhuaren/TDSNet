#This tool aim to solve 2D image semantic segmentation task by 2D PSPNET.

#data structure

data: Challenge data

# Install 

1. Highly recommend you install Anaconda3

#cd to MASK_RCNN folder
conda install pytorch=0.4.1 cuda90 -c pytorch
pip install albumentations=0.0.4
pip install tqdm
conda install -c pytorch torchvision
pip install Pillow==6.1
pip install -U scikit-learn
#pip install opencv-python

# Utilize

1. training:

sh train.sh

2. testing:

sh recolor.sh




