from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import os
import util.config

def main():
    


    instrument_index = os.listdir(train_path)
    
    for sub_index in instrument_index:
        
        instrument_folder = sub_index
        mask_folder = train_path / instrument_folder / 'labels'
        
        (cropped_train_path / instrument_folder / 'images').mkdir(exist_ok=True, parents=True)
        instrument_mask_folder = (cropped_train_path / instrument_folder / 'instruments_masks')
        instrument_mask_folder.mkdir(exist_ok=True, parents=True)

        for file_name in tqdm(list((train_path / instrument_folder / 'left_frames').glob('*'))):
            
            try:
                img = cv2.imread(str(file_name))
            except:
                continue

            try:
                mask = cv2.imread(str(mask_folder / file_name.name))
            except:
                continue
            
            img = img[:height, :width]
            cv2.imwrite(str(cropped_train_path / instrument_folder / 'images' / file_name.name), img)

            mask_instruments = np.zeros((height, width))
            for mask_label, sub_color in enumerate(class_color):
                temp = np.logical_and(np.logical_and(mask[:,:,0] == sub_color[2], mask[:,:,1] == sub_color[1]),
                                      mask[:,:,2] == sub_color[0])
                mask_instruments[temp] = mask_label 
            mask_instruments = (mask_instruments[:height, :width]).astype(np.uint8)
            cv2.imwrite(str(instrument_mask_folder / file_name.name), mask_instruments)
            
if __name__ == '__main__':
    main()

