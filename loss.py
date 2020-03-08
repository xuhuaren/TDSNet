import torch
from torch import nn
from torch.nn import functional as F
import utils
import numpy as np


def calculate_l1_loss(output, target, lagrange_coef=0.0005):
    l1_crit = nn.L1Loss(size_average=False)  # SmoothL1Loss
    reg_loss = l1_crit(output.argmax(dim=1).float(), target.float())

    return lagrange_coef * reg_loss



class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        targets_de=torch.argmax(targets, dim=1) 
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets_de)
        

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = targets[:, cls].float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum()
#                loss += (1-(2*intersection + eps) / (union + eps)) * self.jaccard_weight
                loss += (1-(intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
#                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss
    
class LossMultiRE:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        targets_de=torch.argmax(targets, dim=1) 
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets_de)
        

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):

                jaccard_target = targets[:, cls].float()
                jaccard_output = outputs[:, cls].exp()
                
                if jaccard_target.sum() == 0:
                    if jaccard_output.sum() == 0:
                        loss +=0
                    else:
                        loss +=1 * self.jaccard_weight
                else:
                    intersection = (jaccard_output * jaccard_target).sum()
                    union = jaccard_output.sum() + jaccard_target.sum()
                    loss += (1-(intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
#                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss

