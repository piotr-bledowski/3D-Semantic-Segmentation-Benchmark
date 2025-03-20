import torch
import os
from tqdm import tqdm

path = './data/s3dis'

labels = []

total_areas = 0

for area in os.listdir(path):
    if '.DS' not in area and os.path.isdir(os.path.join(path, area)):
        total_areas += 1

i = 0

for area in os.listdir(path):
    if '.DS' not in area and os.path.isdir(os.path.join(path, area)):
        rooms = []
        for room in tqdm(os.listdir(os.path.join(path, area))):
            if '.DS' not in room and os.path.isdir(os.path.join(path, area, room)): # .DS_Store is a hidden file, an artifact that popped up for some reason
                x = []
                y = []
                for obj in os.listdir(os.path.join(path, area, room, 'Annotations')):
                    if '.DS' not in obj:
                        label = obj.split('_')[0]
                        if label not in labels:
                            labels.append(label)

                        with open(os.path.join(path, area, room, 'Annotations', obj), 'r') as f:
                            for line in f.readlines():
                                x_tmp = [float(z) for z in line.split(' ') if 'x' not in z]
                                if len(x_tmp) == 6:
                                    x.append(x_tmp)
                                    y.append(label)
                if x:
                    room_data = {'x': x, 'y': y}
                    rooms.append(room_data)
        i += 1
        print(f'{i} / {total_areas}')
        torch.save(rooms, f'./data/s3dis{i}.pt')
