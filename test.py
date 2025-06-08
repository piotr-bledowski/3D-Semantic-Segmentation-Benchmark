import pickle

d = pickle.load(open('data_chunked/chunked_s3dis_index_mapping.pkl', 'rb'))

# print(d)

import torch


# print(type(d))

# print(type(d[0]))

# print(d['x'].shape)

y = []
for j in range(1, 10):
    d = torch.load(f'data_chunked/s3dis6_chunk{j}.pt')

    for i in list(set(d[0]['y'])):
        if i not in y:
            y.append(i)

print(y)

print(len(y))

