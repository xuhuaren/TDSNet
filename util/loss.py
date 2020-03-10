import torch
from torch import nn
from torch.nn import functional as F
import util.utils as utils
import numpy as np


def calculate_l1_loss(output, target, lagrange_coef=0.0005):
    l1_crit = nn.L1Loss(size_average=False)  # SmoothL1Loss
    reg_loss = l1_crit(output.argmax(dim=1).float(), target.float())

    return lagrange_coef * reg_loss

   
class LossHybrid:
    
      
    """
    Our proposed hybrid loss, cross-entropy combined with iou loss
    Attributes:
        jaccard_weight: iou loss weight
        num_classes: segment class numbers
    """  
    
    
    def __init__(self, jaccard_weight=0.5, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        
        targets_out=torch.argmax(targets, dim=1) 
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets_out)
        
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
        return loss

