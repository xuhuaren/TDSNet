"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
from prepare_train_val import get_split
from dataset import RoboticsDataset
import cv2
from models import UNet16, LinkNet34, UNet11, UNet, AlbuNet, DUNet16,DUNet11
from models_new import DinkNet34, DinkNet50, DinkNet101
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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)


def get_model(model_path, model_type='UNet11', problem_type='binary'):
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


def predict(model, from_file_names, batch_size, to_path, problem_type, img_transform):
    loader = DataLoader(
        dataset=RoboticsDataset(from_file_names, transform=img_transform, mode='predict', problem_type=problem_type),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = utils.cuda(inputs)

            outputs,_,_,_ = model(inputs)

            for i, image_name in enumerate(paths):
                if problem_type == 'binary':
                    factor = prepare_data.binary_factor
                    t_mask = (F.sigmoid(outputs[i, 0]).data.cpu().numpy() * factor).astype(np.uint8)
                elif problem_type == 'parts':
                    factor = prepare_data.parts_factor
                    t_mask = (outputs[i].data.cpu().numpy().argmax(axis=0) * factor).astype(np.uint8)
                elif problem_type == 'instruments':
                    factor = prepare_data.instrument_factor
                    t_mask = (outputs[i].data.cpu().numpy().argmax(axis=0) * factor).astype(np.uint8)

                h, w = t_mask.shape

                full_mask = np.zeros((original_height, original_width))
                full_mask[h_start:h_start + h, w_start:w_start + w] = t_mask

                instrument_folder = Path(paths[i]).parent.parent.name

                (to_path / instrument_folder).mkdir(exist_ok=True, parents=True)

                cv2.imwrite(str(to_path / instrument_folder / (Path(paths[i]).stem + '.png')), full_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.5, type=float)
    arg('--model_path', type=str, default='data/models/UNet', help='path to model folder')
    arg('--model_type', type=str, default='UNet', help='network architecture')
    arg('--output_path', type=str, help='path to save images', default='1')
    arg('--batch-size', type=int, default=2)
    arg('--fold', type=int, default=-1, help='-1: all folds')
    arg('--problem_type', type=str, default='binary', choices=['binary', 'parts', 'instruments'])
    arg('--workers', type=int, default=12)


    args = parser.parse_args()

    if args.fold == -1:
        for fold in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]:
            _, file_names = get_split(fold)
            model = get_model(str(Path(args.model_path).joinpath('model_{fold}_{net}_{we}.pt'.format(fold=fold,net=args.model_type,we=args.jaccard_weight))),
                          model_type=args.model_type, problem_type=args.problem_type)

            print('num file_names = {}'.format(len(file_names)))

            output_path = Path(args.output_path)
            output_path.mkdir(exist_ok=True, parents=True)

            predict(model, file_names, args.batch_size, output_path, problem_type=args.problem_type,
                    img_transform=img_transform(p=1))
    else:
        _, file_names = get_split(args.fold)
        model = get_model(str(Path(args.model_path).joinpath('model_{fold}_{net}_{we}_stage1.pt'.format(fold=args.fold,net=args.model_type,we=args.jaccard_weight))),
                          model_type=args.model_type, problem_type=args.problem_type)

        print('num file_names = {}'.format(len(file_names)))

        output_path = Path(args.output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        predict(model, file_names, args.batch_size, output_path, problem_type=args.problem_type,
                img_transform=img_transform(p=1))
