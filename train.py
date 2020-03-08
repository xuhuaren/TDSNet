import argparse
import torch
from pathlib import Path
from validation import validation_multi
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from models import DUNet16
from loss import LossMultiRE
from dataset import RoboticsDataset
import utils
from prepare_train_val import get_split
from albumentations import (HorizontalFlip,
                            VerticalFlip,
                            Normalize,
                            Compose)


def main():
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--alpha', default=0.2, type=float)
    arg('--beta', default=0.6, type=float)
    arg('--gama', default=0.2, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=6)
    arg('--model', type=str, default='DUNet16')
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 12 

    if args.model == 'DUNet16':
        model = DUNet16(num_classes=num_classes, pretrained=True)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    cudnn.benchmark = True        

    loss = LossMultiRE(num_classes=num_classes, jaccard_weight=args.jaccard_weight)


    def make_loader(file_names, shuffle=False, transform=None, problem_type='instruments', batch_size=1):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.fold)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

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

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1), problem_type=args.type,
                               batch_size=args.batch_size)
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=args.type,
                               batch_size=len(device_ids))
    


        


    if args.type == 'binary':
        valid = validation_binary
    else:
        valid = validation_multi

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes
    )


if __name__ == '__main__':
    main()
