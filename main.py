# In[Import]
import time
import random
import argparse
import operator
import numpy as np

import os
import sys
import os.path as op
from tqdm import tqdm
from pathlib import Path
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_isolated_nodes

from GNN import BA3Net
from datasets import ba3motif
# from gnn.mutgnn import *
# from gnn.hivgnn import *
# from utils.seed import GlobalSeed
# from utils.train import get_dataset
# from utils.math import Harmonic
# from train.config import root
from vinfor_v2 import vinfor
# from utils.noise import add_noise

# from datasets.union_dataset import UnionDataset

# In[Settings]
np.set_printoptions(precision = 6, suppress = True)
ClassesDict = {'mut': 2, 'mnist': 10, 'ba3': 3, 'vg': 5,
               'hiv': 2, 'ppa': 37}

def GlobalSeed(seed,opt = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = opt
    torch.backends.cudnn.deterministic = not opt

# In[argparse]
def Args():
    Parser = argparse.ArgumentParser(description = "vinfor")
    
    Parser.add_argument('--data_root', type = str, default = './data')
    Parser.add_argument('--model_path', type = str, default = './model')
    Parser.add_argument('--dataset', type = str, default = 'ba3', help = 'Dataset to train.')
    Parser.add_argument('--cuda', type = int, default = 0, help = 'GPU device.')
    Parser.add_argument('--vae_epoch', type = int, default = 10, help = 'Number of auto-encoder train epoch.')
    Parser.add_argument('--exp_epoch', type = int, default = 0, help = 'Number of explainer train epoch.')
    Parser.add_argument('--epoch', type = int, default = 10, help = 'Number of integral model train epoch.')
    Parser.add_argument('--vae_hidden', type = int, default = 64, help = 'hidden dim of Vae.')
    Parser.add_argument('--vae_out', type = int, default = 16, help = 'out dim of Vae.')
    Parser.add_argument('--alpha_hidden', type = int, default = 64, help = 'hidden dim of AlphaNet.')
    Parser.add_argument('--vae_lr', type = float, default = 1e-5, help = 'Auto-Encoder learning rate.')
    Parser.add_argument('--alpha_lr', type = float, default = 1e-6, help = 'Alpha-Net learning rate.')
    Parser.add_argument('--lr', type = float, default = 1e-5, help = 'Global learning rate.')
    Parser.add_argument('--ratio', type = float, default = 0, help = 'Noise ratio.')
    Parser.add_argument('--tau', type = float, default = 0.3, help = 'Temperature in gumbel-softmax distribution.')
    Parser.add_argument('--gamma', type = float, default = 1, help = 'Hyper param between FidLoss and CTSLoss.')
    Parser.add_argument('--batchsize', type = int, default = 1, help = 'Batchsize.')
    Parser.add_argument('--seed', type = int, default = 1, help = 'Random seed.')
    Parser.add_argument('--test', type = int, default = 2, help = 'Interval of test.')
    Parser.add_argument('--result', type = str, default = './results/VInfoR/', help = 'Result directory.')
    
    return Parser.parse_args()

args = Args()
# args.dataset = 'mut'
# args.cuda = 0
GlobalSeed(args.seed)

# In[Dataset]
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.cuda}')
else:
    device = torch.device('cpu')

# Raw dataset
# train, valid, test = get_dataset(args.dataset, root['data'])
train, valid, test = ba3motif(args.data_root, 'train'), ba3motif(args.data_root, 'valid'), ba3motif(args.data_root, 'test')

# Attacked dataset
# adv_train, adv_valid, adv_test = get_dataset('adv' + args.dataset, root['data'])

# Dimension of edge attribute.
edge_dim = train[0].edge_attr.size(1)
# edge_dim = adv_train[0].edge_attr.size(1)
# print('edge dim:',edge_channel)

# Dimension of node attribute.
node_dim = torch.flatten(train[0].x, 1, -1).size(1)
# node_dim = torch.flatten(adv_train[0].x,1,-1).size(1)
# print('node dim:',node_channel)

# gnn = op.join(root['param'],'gnns/%snet.pt' % args.dataset)

'''Save vinfor.'''
# save_path = op.join(root['param'],'VInfoR/')
# mode = 'atk'
# save_path += mode

os.makedirs(args.result, exist_ok = True)

'''
# def DataLoader.__init__(self,dataset: Union[Dataset, List[BaseData]],
#              batch_size: int = 1,shuffle: bool = False)
'''
trainloader = DataLoader(train, shuffle = True)
validloader = DataLoader(valid, shuffle = True)
testloader = DataLoader(test, shuffle = False)

# trainloader = DataLoader(adv_train, shuffle = True)
# validloader = DataLoader(adv_valid, shuffle = True)
# testloader = DataLoader(adv_test, shuffle = False)

Loader = [trainloader, validloader, testloader]

# In[Random Choose]
def choose(loader):
    data = next(iter(loader))
    return data

def mask_to_index(mask):
    return mask.nonzero(as_tuple = False).view(-1)

# In[vinfor]
vinfor = vinfor(
    device = device, 
    target_model_path = args.model_path,
    node_dim = node_dim,
    edge_dim = edge_dim,
    vae_hidden = args.vae_hidden,
    vae_out = args.vae_out,
    alpha_hidden = args.alpha_hidden,
    Classes = ClassesDict[args.dataset]
)

vae_para = vinfor.vae.parameters()
alpha_para = vinfor.alpha_net.parameters()

opt = torch.optim.Adam(
    [
        {'params':vae_para, 'lr':args.vae_lr},
        {'params':alpha_para, 'lr':args.alpha_lr}
    ],
    args.lr
)

'''opt for optimizer, sdl for scheduler'''
# opt = torch.optim.Adam(parameters,args.lr)
sdl = ReduceLROnPlateau(
    opt, 
    mode = 'min', 
    factor = 0.5,
    patience = 3,
    min_lr = 1e-6
)

# In[Vae]
# Vae warm-up.
# ----------------------------------------------------------------- #
for epoch in range(1, args.vae_epoch + 1):
    loss = 0
    for g in Loader[0]:
        g.to(device)
        if g.num_nodes != torch.unique(g.edge_index).shape[0]:
            edge_index, edge_attr, mask = \
                remove_isolated_nodes(g.edge_index, 
                                      g.edge_attr, 
                                      num_nodes = g.num_nodes)
            non_isolated = mask_to_index(mask)
            g.x = g.x[non_isolated]
            g.edge_index = edge_index
            g.edge_attr = edge_attr
        # g[0].to(device)
        # g[1].to(device)

        VaeLoss = vinfor.vae_train(g)
        loss += VaeLoss.item()
        opt.zero_grad()
        VaeLoss.backward()
        opt.step()
    
    loss /= len(Loader[0].dataset)
    print('Vae epoch:%d' % epoch + 
          ', vae train loss:%.4f' % loss)
    
    valid_loss = 0
    with torch.no_grad():
        # L = len(Loader[1].dataset)
        for g in Loader[1]:
            g.to(device)
            if g.num_nodes != torch.unique(g.edge_index).shape[0]:
                edge_index, edge_attr, mask = \
                    remove_isolated_nodes(g.edge_index, 
                                          g.edge_attr,
                                          num_nodes = g.num_nodes)
                non_isolated = mask_to_index(mask)
                g.x = g.x[non_isolated]
                g.edge_index = edge_index
                g.edge_attr = edge_attr
            # g[0].to(device)
            # g[1].to(device)

            VaeLoss = vinfor.vae_train(g)
            valid_loss += VaeLoss.item()
        valid_loss /= len(Loader[1].dataset)
        
        # print('len:',L)
        # valid_loss /= L
        
        print('Vae epoch:%d' % epoch + 
              ', vae valid loss:%.4f' % valid_loss)
# ----------------------------------------------------------------- #

# In[Jointly]
min_val_loss = None
'''Loader: train, valid, test.'''
for epoch in range(1, args.epoch + 1):
    loss_sum = 0
    train_vae_loss = 0
    train_cts_loss = 0
    train_fid_loss = 0
    train_i = 1
    # ------------ Start Training ------------ #
    for g in Loader[0]:
        g.to(device)
        if g.num_nodes != torch.unique(g.edge_index).shape[0]:
                edge_index, edge_attr, mask = \
                    remove_isolated_nodes(g.edge_index,
                                          g.edge_attr,
                                          num_nodes = g.num_nodes)
                non_isolated = mask_to_index(mask)
                g.x = g.x[non_isolated]
                g.batch = g.batch[non_isolated]
                g.edge_index = edge_index
                g.edge_attr = edge_attr
        
        # print('g.x.shape:',g.x.shape,
        #       'g.batch.shape:',g.batch.shape)
        
        VaeLoss, FidLoss = vinfor.pretrain((g,))
        # print('success train', train_i)
        # train_i = train_i + 1
        
        loss = args.gamma * FidLoss + VaeLoss
        loss_sum += loss.item()
        train_vae_loss += VaeLoss.item()
        train_fid_loss += FidLoss.item()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    loss_sum /= len(Loader[0].dataset)
    train_vae_loss /= len(Loader[0].dataset)
    train_fid_loss /= len(Loader[0].dataset)
    print('Epoch: %d ' % epoch +
          'Train Loss: %.4f' % loss_sum)
    
    print('VaeLoss: %.4f ' % train_vae_loss + 
          'FidLoss: %.4f' % train_fid_loss)
        
    # ------------ End Training ------------ #
    
    val_loss_sum = 0
    val_vae_loss = 0
    val_fid_loss = 0
    val_cts_loss = 0
    
    valid_i = 1
    # ------------ Start Validation ------------ #
    with torch.no_grad():
        for g in Loader[1]:
            
            g.to(device)
            g_y = vinfor.model(g).argmax().item()
            # print('g.y:',g_y)
            if g.num_nodes != torch.unique(g.edge_index).shape[0]:
                edge_index, edge_attr, mask = \
                    remove_isolated_nodes(g.edge_index, 
                                          g.edge_attr,
                                          num_nodes = g.num_nodes)
                non_isolated = mask_to_index(mask)
                g.x = g.x[non_isolated]
                g.batch = g.batch[non_isolated]
                g.edge_index = edge_index
                g.edge_attr = edge_attr
            
            VaeLoss, FidLoss = vinfor.pretrain((g,))
            # print('success valid ', valid_i)
            # valid_i = valid_i + 1
            
            val_loss = args.gamma * FidLoss + VaeLoss
            
            val_vae_loss += VaeLoss.item()
            val_fid_loss += (args.gamma * FidLoss).item()
            val_loss_sum += val_loss.item()
            
            if min_val_loss is None or min_val_loss > val_loss:
                min_val_loss = val_loss
                torch.save(vinfor,op.join(save_path,f'{args.dataset}.pt'))
                
        sdl.step(val_loss_sum)
        lr = sdl.optimizer.param_groups[0]['lr']

    val_loss_sum /= len(Loader[1].dataset)
    val_vae_loss /= len(Loader[1].dataset)
    val_fid_loss /= len(Loader[1].dataset)
    val_cts_loss /= len(Loader[1].dataset)
    
    print('Epoch: %d ' % epoch +
          'LR: %.5f ' % lr +
          'Val Loss: %.4f' % val_loss_sum)
    
    print('VaeLoss: %.4f ' % val_vae_loss +
          'FidLoss: %.4f' % val_fid_loss)
    
    # ------------ End Validation ------------ #
    
    # ------------ Start Testing ------------ #
    if epoch % args.test == 0 :
        Metric = {'Fdl':[],'Ps':[],'Pn':[],'Fns':[]}
        
        pM = []
        result = {}
        test_i = 1
        for g in Loader[2]:
            # Remove isolated nodes.
            if g.num_nodes != torch.unique(g.edge_index).shape[0]:
                edge_index, edge_attr, mask = \
                    remove_isolated_nodes(g.edge_index,
                                          g.edge_attr,
                                          num_nodes = g.num_nodes)
                non_isolated = mask_to_index(mask)
                g.x = g.x[non_isolated]
                g.batch = g.batch[non_isolated]
                g.edge_index = edge_index
                g.edge_attr = edge_attr
                
            # Add noise if needed.
            if args.ratio != 0:
                g = add_noise(g,args.ratio)
            g.to(device)
            # g[0].to(device)
            # g[1].to(device)
            explain = vinfor.explain(g, finetune = True)
            pM.extend(explain.reshape(-1).tolist())
            
            Fidelity = vinfor.Fidelity()
            Ps, Pn = vinfor.FnsScore()
            
            # print('success test:',test_i)
            # test_i = test_i + 1
            
            
            Metric['Fdl'].append(Fidelity)
            Metric['Ps'].append(Ps)
            Metric['Pn'].append(Pn)
            
        result['Fdl'] = np.array(Metric['Fdl']).mean(axis=0).tolist()
        result['Prob-S'] = np.array(Metric['Ps']).mean(axis=0).tolist()
        result['Prob-N'] = np.array(Metric['Pn']).mean(axis=0).tolist()
        result['Fns'] = Harmonic(result['Prob-S'],result['Prob-N'])
        
        metrics = {'Fdl':result['Fdl'],
                   'Ps:':result['Prob-S'],
                   'Pn':result['Prob-N'],
                   'Fns':result['Fns']}

        print('Fidelity: %.4f' % result['Fdl'],
              '\nProb-S: %.4f' % result['Prob-S'],
              '\nProb-N: %.4f' % result['Prob-N'],
              '\nFns Score: %.4f' % result['Fns'],
              '\nMean Probability: %.4f\n' % np.array(pM).mean())
        
        # os.makedirs(args.result,exist_ok = True)
        
        # with open(op.join(args.result,f'{args.dataset}.json'),'w') as f:
        #     json.dump(result,f,indent = 4)
    # ------------ End Testing ------------ #