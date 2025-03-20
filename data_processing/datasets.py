import torch
from torch.utils.data import Dataset
import os


class S3DISDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.x = []
        self.y = []
        self.labels = []
        rooms90088 = []
        for area in os.listdir(path):
            for room in os.listdir(os.path.join(path, area)):
                if room != '.DS_Store' and os.path.isdir(os.path.join(path, area, room)): # .DS_Store is a hidden file, an artifact that popped up for some reason
                    for obj in os.listdir(os.path.join(path, area, room, 'Annotations')):
                        label = obj.split('_')[0]
                        if label not in self.labels:
                            self.labels.append(label)

                        with open(os.path.join(path, area, room, 'Annotations', obj), 'r') as f:
                            for line in f.readlines():
                                self.x.append(float(x) for x in line.split(' '))
                                self.y.append(label)
                

                torch.save()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    dataset = S3DISDataset('./data/s3dis')
    print(len(dataset))
    print(dataset.labels)
