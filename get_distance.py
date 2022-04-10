from Dataloader import load_data
from utils import shortest_path
from tqdm import tqdm
import torch
import os
import time

if __name__ == "__main__":
#     choices = ['ACTOR',
# 'TEXAS',
# 'WISCONSIN',
# 'CORNELL',
# 'AmazonComputers',
# 'Citeseer',
# 'CoauthorCS',
# 'AmazonPhoto',
# 'Pubmed',
# 'Cora',
# 'CoauthorPyhysics']
    choices = ['squirrel']
    root = 'data/'

    t = time.time()
    for choice in tqdm(choices):
        data = load_data(choice, 0)
        distance = shortest_path(data.x.size(0), data.edge_index)
        torch.save(distance, os.path.join(root, choice, 'distance.pth'))
    print(f"final time: {time.time() - t}")

