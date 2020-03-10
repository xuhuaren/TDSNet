import numpy as np
import util.utils as utils
from torch import nn
import torch
from util.config import *

def general_dice(y_true, y_pred):
    
    result = []
    for instrument_id in range(num_classes):
        
        if np.all( y_true != instrument_id ):
            if np.all( y_pred != instrument_id ):
                result += [1]
            else:
                result += [0]
        else:
            result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_jaccard(y_true, y_pred):
    result = []


    for instrument_id in range(num_classes):
        if np.all( y_true != instrument_id ): 
            if np.all( y_pred != instrument_id ):
                result += [1]
            else:
                result += [0]
        else:
            result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def validation(model: nn.Module, criterion, valid_loader, num_classes):
    
    """
    validation metric including dice and iou

    Args:
        model: load model
        criterion: measure metric
        valid_loader: validation data loader
        num_classes: class task
        
    Returns:
        valid_loss: valid loss calculate by gpu
        average_iou: iou score calculate by cpu    
    Raises:
        None
    """ 
    
    with torch.no_grad():
        model.eval()
        losses = []
        dice=[]
        iou=[]
        for inputs, targets,_ in valid_loader:
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)
            outputs,_,_ = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            output_classes = outputs.data.cpu().numpy().argmax(axis=1)
            target_classes = targets.data.cpu().numpy().argmax(axis=1)
            dice += [general_dice(target_classes, output_classes)]
            iou += [general_jaccard(target_classes, output_classes)]

        valid_loss = np.mean(losses)
        average_iou = np.mean(iou)
        average_dices = np.mean(dice)

        print('Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.4f}'.format(valid_loss, average_iou, average_dices))

        return [valid_loss, average_iou]



