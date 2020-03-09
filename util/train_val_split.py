import os
from pathlib import Path
import glob

def get_split(fold):
    folds = {0: ['seq_1', 'seq2', 'seq3', 'seq4']}
    
    data_path = Path('../data')
    train_path = data_path / 'cropted_train'
    
    train_file_names = []
    val_file_names = []
    ids = os.listdir(train_path)
    
    for instrument_id in ids:
        if instrument_id in folds[fold]:
            val_file_names += list((train_path / (instrument_id) / 'images').glob('*'))
        else:
            train_file_names += list((train_path / (instrument_id) / 'images').glob('*'))

    return train_file_names, val_file_names