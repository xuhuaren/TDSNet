import numpy as np
import utils
from torch import nn
import torch

def general_dice(y_true, y_pred):
    result = []


    for instrument_id in range(1,12):
        
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


    for instrument_id in range(1,12):
        
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


def validation_binary(model: nn.Module, criterion, valid_loader, num_classes=None):
    with torch.no_grad():
        model.eval()
        losses = []

        jaccard = []

        for inputs, targets in valid_loader:
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            jaccard += [jaccard(targets, (outputs > 0).float()).item()]

        valid_loss = np.mean(losses)  # type: float

        valid_jaccard = np.mean(jaccard).astype(np.float64)

        print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
        metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard}
        return metrics




def validation_multi(model: nn.Module, criterion, valid_loader, num_classes):
    with torch.no_grad():
        model.eval()
        losses = []
        dice=[]
        iou=[]

        for inputs, targets,_ in valid_loader:
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)
            outputs,_,_,_ = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            output_classes = outputs.data.cpu().numpy().argmax(axis=1)
            target_classes = targets.data.cpu().numpy().argmax(axis=1)
            dice += [general_dice(target_classes, output_classes)]
            iou += [general_jaccard(target_classes, output_classes)]

       
        valid_loss = np.mean(losses)
        average_iou = np.mean(iou)
        average_dices = np.mean(dice)

        print(
            'Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.4f}'.format(valid_loss, average_iou, average_dices))
#        metrics = {'valid_loss': valid_loss, 'iou': average_iou}
#        metrics.update(valid_loss)
#        metrics.update(average_iou)
        return [valid_loss, average_iou]



