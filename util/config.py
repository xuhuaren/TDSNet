from pathlib import Path

data_path = Path('./data')
train_path = data_path / 'raw'
cropped_train_path = data_path / 'cropted_train'
    
height, width = 1024, 1280
class_color=[[0,0,0], [0,255,0], [0,255,255], [125,255,12],
            [255,55,0], [24,55,125], [187,155,25], [0,255,125], [255,255,125],
            [123,15,175], [124,155,5], [12,255,141]]