from pathlib import Path

#data main path
data_path = Path('./data')
#raw/original data saved path
train_path = Path('/mnt/data/home/renxuhua/temp')
#post-processing data saved path
cropped_train_path = data_path / 'cropted_train'

#saved image height, width 
height, width = 1024, 1280
#robot18 each class color transform
class_color=[[0,0,0], [0,255,0], [0,255,255], [125,255,12],
            [255,55,0], [24,55,125], [187,155,25], [0,255,125], [255,255,125],
            [123,15,175], [124,155,5], [12,255,141]]
# class task number
num_classes = 12 
# scene task number
num_scenes = 3

#scene task class, three classes scene_0: preparing,
#scene_1: peeling, scene_2:incising 
scene_0 = ['seq_1','seq_2','seq_4','seq_5','seq_7']
scene_1 = ['seq_3','seq_10','seq_11','seq_12','seq_16']
scene_2 = ['seq_6','seq_9','seq_13','seq_14','seq_15']
