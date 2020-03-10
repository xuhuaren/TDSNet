import argparse
import torch
from pathlib import Path
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import util.utils as utils
from albumentations import (HorizontalFlip,
                            VerticalFlip,
                            Normalize,
                            Compose)
from util.validation import validation
from util.models import TDSNet
from util.loss import LossHybrid
from util.dataset import RoboticsDataset
from util.train_val_split import get_split
from util.config import *

def make_loader(file_names, shuffle=False, transform=None, batch_size=1, works = 4):
    return DataLoader(
        dataset=RoboticsDataset(file_names, transform=transform),
        shuffle=shuffle,
        num_workers=works,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
    
def train_transform(p=1):
    return Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        Normalize(p=1)
    ], p=p)

def val_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)

def main():
    
    """
    Training script which could obtain trained model

    Args:
        some parameters saved in ./util/config.py
        alpha: weight of class task
        beta: weight of sync task
        gama: weight of scene task
        device-ids: the device id in computer
        fold: the validation set id
        root: model saved path
        batch-size: batch size
        n-epochs: maximum epochs
        lr: learning rate
        workers: cpu kernal numbers
        model: model which utilized
        num_classes: class/segment task numbers; num_scenes class numbers saved in ./util/config.py
        
    Returns:
        model saved in root path.
    Raises:
        None
    """  
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--alpha', default=0.2, type=float)
    arg('--beta', default=0.4, type=float)
    arg('--gama', default=0.2, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=6)
    arg('--model', type=str, default='TDSNet')
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    if args.model == 'TDSNet':
        model = TDSNet(num_classes = num_classes, num_scenes = num_scenes, pretrained=True)
    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    cudnn.benchmark = True        

    loss = LossHybrid(num_classes=num_classes)

    train_file_names, val_file_names = get_split(args.fold)
    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))
    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1), 
                               batch_size=args.batch_size)
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), 
                               batch_size=args.batch_size)
    

    valid = validation

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes,
        weights = [args.alpha, args.beta, args.gama]
    )


if __name__ == '__main__':
    main()
