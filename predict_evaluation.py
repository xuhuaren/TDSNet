"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
from prepare_train_val import get_split
from dataset import RoboticsDataset
import cv2
from models import UNet16, LinkNet34, UNet11, UNet, AlbuNet, DUNet16,DUNet11
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
import prepare_data
from torch.utils.data import DataLoader
from torch.nn import functional as F
from prepare_data import (original_height,
                          original_width,
                          h_start, w_start
                          )
from albumentations import Compose, Normalize
from sklearn.metrics import accuracy_score

train_1=['seq_1','seq_2','seq_4','seq_5','seq_7']
train_2=['seq_3','seq_10','seq_11','seq_12','seq_16']
train_3=['seq_6','seq_9','seq_13','seq_14','seq_15']

def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)


def get_model(model_path, model_type='DUNet16', problem_type='instruments'):
    """

    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34', 'AlbuNet'
    :param problem_type: 'binary', 'parts', 'instruments'
    :return:
    """
    if problem_type == 'binary':
        num_classes = 1
    elif problem_type == 'parts':
        num_classes = 4
    elif problem_type == 'instruments':
        num_classes = 12

    if model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
    elif model_type == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif model_type == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes)
    elif model_type == 'AlbuNet':
        model = AlbuNet(num_classes=num_classes)
    elif model_type == 'UNet':
        model = UNet(num_classes=num_classes)
    elif model_type == 'DinkNet34':
        model = DinkNet34(num_classes=num_classes)
    elif model_type == 'DinkNet50':
        model = DinkNet50(num_classes=num_classes)
    elif model_type == 'DinkNet101':
        model = DinkNet101(num_classes=num_classes)
    elif model_type == 'DUNet16':
        model = DUNet16(num_classes=num_classes)
    elif model_type == 'DUNet11':
        model = DUNet11(num_classes=num_classes)
        
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model


def predict(model, from_file_names, batch_size, problem_type, img_transform):
    
    loader = DataLoader(
        dataset=RoboticsDataset(from_file_names, transform=img_transform, mode='train', problem_type=problem_type),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )
    class_res = []
    scene_res = []
    class_gt = []
    scene_gt = []

    with torch.no_grad():
        for batch_num, (inputs, targets, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = utils.cuda(inputs)

            outputs,scene,class_0,class_1 = model(inputs)

            for i, image_name in enumerate(paths):

                scene_mask = (scene[i].data.cpu().numpy().argmax(axis=0)).astype(np.uint8)                
                class_mask = (class_0[i].data.cpu().numpy()).astype(np.bool).astype(np.uint8)  
                instrument_folder =paths[i].split('/')[1]
                
                exist_class = [1 if targets[0, c,:,:].sum()>0 else 0 for c in range(12)]
                class_gt.append(exist_class)
                class_res.append(class_mask)    
                
                if instrument_folder in train_1:
                    scene_gt.append(0)
                elif instrument_folder in train_2:
                    scene_gt.append(1)
                elif instrument_folder in train_3:
                    scene_gt.append(2)
                    
                scene_res.append(scene_mask)
                
                

                
    acc = accuracy_score(scene_res, scene_gt)
    print("scene Acc:" + str(acc))
    

    ave = []
    for sub_ in range(12):
        temp_res = []
        temp_gt = []
        for idx, sub_class in enumerate(class_gt):
            #print(class_gt)
            #print(sub_class)
         
            temp_gt.append(class_gt[idx][sub_])
            temp_res.append(class_res[idx][sub_])
        #print(temp_gt)
        #print(temp_res)
        #assert(0)
        ave.append(accuracy_score(temp_res, temp_gt))


    print("Class Acc:" + str(np.mean(ave)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models/UNet', help='path to model folder')
    arg('--model', type=str, default='UNet', help='network architecture')
    arg('--output_path', type=str, help='path to save images', default='1')
    arg('--batch-size', type=int, default=2)
    arg('--fold', type=int, default=1, help='-1: all folds')
    arg('--problem_type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])
    arg('--workers', type=int, default=12)


    args = parser.parse_args()

    _, file_names = get_split(args.fold)
    model = get_model(args.model_path)

    print('num file_names = {}'.format(len(file_names)))

    predict(model, file_names, args.batch_size, problem_type=args.problem_type,
                img_transform=img_transform(p=1))
