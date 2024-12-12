'''
Dateset: BA-3Motif
Create InMemoryDataset.
'''
# In[Import]
import os
import random
import numpy as np
import pickle as pkl
import os.path as op

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
# In[BA-3motif]
class ba3motif(InMemoryDataset):
    # Split dataset to 3 parts.
    splits = ['train','valid','test']
    
    def __init__(self, root, mode = 'test', transform = None, pre_transform = None, pre_filter = None):
        assert mode in self.splits
        self.mode = mode
        
        super().__init__(root,transform,
                         pre_transform,pre_filter)
        
        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.paths = [root + r'/processed/train.pt',
                      root + r'/processed/valid.pt',
                      root + r'/processed/test.pt']
        self.data, self.slices = torch.load(self.paths[idx])
        
        '''
        In: Train.slices
        Out: 
        defaultdict(dict,
                    {'x': tensor([    0,    15,    30,  ..., 48318, 48334, 48350]),
                     'edge_index': tensor([    0,    24,    46,  ..., 64972, 64990, 65010]),
                     'edge_attr': tensor([    0,    24,    46,  ..., 64972, 64990, 65010]),
                     'y': tensor([   0,    1,    2,  ..., 2198, 2199, 2200]),
                     'Role': tensor([   0,    1,    2,  ..., 2198, 2199, 2200]),
                     'GTMask': tensor([   0,    1,    2,  ..., 2198, 2199, 2200]),
                     'name': tensor([   0,    1,    2,  ..., 2198, 2199, 2200])})
        
        '''
        
    @property
    def raw_file_names(self):
        return ['BA-3motif.npy']

    @property
    def processed_file_names(self):
        return ['train.pt','valid.pt','test.pt']
    
    def download(self):
        pass
        # filename = 'BA-3motif.npy'
        # if not op.exist(op.join(self.raw_dir,'raw',filename)):
        #     print('File does not exist.')
        #     raise FileNotFoundError
            
    def process(self):
        '''
        EdgeID: An array of shape [2,X], X is number of edges.
        Label : Class of whole graph.
        GT: Ground truth for whether each edge belongs to motif(explanation).
        Role: Point out the role of node in motif.
        _ : Placeholder for position.
        '''
        pass
