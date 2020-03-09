import argparse
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
from util.train_val_split import get_split
from util.dataset import RoboticsDataset
from util.models import TDSNet
from util.models import UNet16
from util.config import *
from pathlib import Path
from tqdm import tqdm
from albumentations import Compose
from albumentations import Normalize
import util.utils as utils
from util.validation import general_dice

def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)
    
def get_model(model_path, model_type="TDSNet"):
    
    """
    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34', 'AlbuNet'
    :param problem_type: 'binary', 'parts', 'instruments'
    :return:
    """
    
    if model_type == "TDSNet":
        model = TDSNet(num_classes=num_classes)
    elif model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
        
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()
    model.eval()

    return model


def predict_evaluate(model, from_file_names, args, to_path, img_transform):
    
    loader = DataLoader(
        dataset=RoboticsDataset(from_file_names, transform=img_transform, mode='train'),
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )
    result_dice = []

    with torch.no_grad():
        
        for batch_num, (inputs, gt, paths) in enumerate(tqdm(loader, desc='Predict')):
            
            inputs = utils.cuda(inputs)
            outputs, _, _ = model(inputs)

            for i, image_name in enumerate(paths):

                seg_mask = (outputs[i].data.cpu().numpy().argmax(axis=0)).astype(np.uint8)
                gt_mask = (gt[i]).astype(np.uint8)
                result_dice += [general_dice(gt_mask, seg_mask)]

                instrument_folder = Path(paths[i]).parent.parent.name
                (to_path / instrument_folder).mkdir(exist_ok=True, parents=True)
                
                full_mask = np.zeros((height, width, 3))
                for mask_label, sub_color in enumerate(class_color):
                    full_mask[t_mask==mask_label, 0]=sub_color[2]
                    full_mask[t_mask==mask_label, 1]=sub_color[1]
                    full_mask[t_mask==mask_label, 2]=sub_color[0]
                cv2.imwrite(str(to_path / instrument_folder / (Path(paths[i]).stem + '.png')), full_mask)
                
    print("The validation set with id{} dice score is {}".format(args.fold, np.mean(result_dice)))
                
                
def main():
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='/runs/debug', help='path to model folder')
    arg('--model_type', type=str, default='TDSNet', help='network architecture')
    arg('--output_path', type=str, help='path to save images', default='1')
    arg('--batch-size', type=int, default=1)
    arg('--fold', type=int, default=0, help='-1: all folds')
    arg('--workers', type=int, default=10)
    args = parser.parse_args()

    _, file_names = get_split(args.fold)
    print('num file_names = {}'.format(len(file_names)))  
    
    model = get_model(str(Path(args.model_path).joinpath('model_{fold}_{net}.pt'.format(fold=args.fold, net=args.model_type))))
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    predict_evaluate(model, file_names, args, output_path, img_transform=img_transform(p=1))
    
    
    

if __name__ == '__main__':
    main()
