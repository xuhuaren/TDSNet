import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from albumentations.torch.functional import img_to_tensor
from util.config import *

def make_one_hot(labels, nums = num_classes):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''

    one_hot = np.zeros((nums, labels.shape[0], labels.shape[1]))
    for num in range(nums):
        one_hot[num,:,:] = np.array(labels == num, dtype='int')
        
    return one_hot

class RoboticsDataset(Dataset):
    def __init__(self, file_names: list, to_augment=False, transform=None, mode='train'):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, self.problem_type)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]
        mask=make_one_hot(mask)

        if self.mode == 'train':
            return img_to_tensor(image), torch.from_numpy(mask).long(), str(img_file_name)
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type):
    
    mask_folder = 'instruments_masks'
    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask).astype(np.uint8)
