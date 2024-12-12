'''
The GNN for BA-3Motif.
'''
# In[Import]
import os
import sys
import time
import random
import argparse

import os.path as op

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import ModuleList
from torch.nn import Softmax
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import LEConv
from torch_geometric.nn import BatchNorm
from torch_geometric.loader import DataLoader

# from gnn.overload import overload
from overload import overload
# from utils.seed import GlobalSeed
from datasets import ba3motif
# from utils.train import TrainProxy
# from utils.train import TestProxy
# from datasets.datasetba3 import ba3motif

# In[argparse]
def Args():
    Des = 'Train ba-3motif gnn model.'
    Parser = argparse.ArgumentParser(description = Des)
    
    Parser.add_argument('--datapath', nargs = '?',
                        default = op.join(op.dirname(__file__),
                                          '..',
                                          'data',
                                          'BA3'),
                        help = 'Data path.')
    
    Parser.add_argument('--modelpath', nargs = '?',
                        default = op.join(op.dirname(__file__),
                                          '..',
                                          'param',
                                          'gnns'),
                        help = 'GNN model path.')
    
    Parser.add_argument('--cuda', type = int, default = 0,
                        help = 'GPU device.')
    Parser.add_argument('--epoch', type = int, default = 100,
                        help = 'Number of gnn train epoch.')
    Parser.add_argument('--lr', type = float, default = 1e-3,
                        help = 'Learning rate.')
    Parser.add_argument('--batchsize', type = int, default = 16,
                        help = 'Batchsize.')
    Parser.add_argument('--numlayers', type = int, default = 2,
                        help = 'Number of GNN layers.')
    Parser.add_argument('--testepoch', type = int, default = 10,
                        help = 'Interval of test.')
    
    return Parser.parse_args()

# In[ba3net]
class BA3Net(torch.nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers = layers
        InputDim = 4
        EmbedDim = 64
        
        # Node embedding layer.
        self.embedding = Linear(InputDim,EmbedDim)
        
        self.CONV = ModuleList()
        self.BN = ModuleList()
        self.RELU = ModuleList()
        
        # Convolution layer.
        for i in range(self.layers):
            conv = LEConv(in_channels = 64, 
                          out_channels = 64)
            bn = BatchNorm(in_channels = 64)
            self.CONV.append(conv)
            self.BN.append(bn)
            self.RELU.append(ReLU())
            
        # Classifier.
        self.relu = ReLU()
        self.Linear1 = Linear(64,64)
        self.Linear2 = Linear(64,3)
        self.softmax = Softmax(dim = 1)

    @overload
    def forward(self,x,EdgeID,EdgeAttr,batch):
        GraphX = self.GraphRep(x,EdgeID,EdgeAttr,batch)
        return self.Predict(GraphX)
    
    @overload
    def NodeRep(self,x,EdgeID,EdgeAttr,batch):
        x = self.embedding(x)
        x = F.dropout(x, p = 0.4)
        for (conv,bn,relu) in zip(self.CONV,self.BN,self.RELU):
            x = conv(x,EdgeID,EdgeAttr)
            x = bn(x)
            x = relu(x)
        x = F.dropout(x, p = 0.4)
        return x
    
    @overload
    def GraphRep(self,x,EdgeID,EdgeAttr,batch):
        NodeX = self.NodeRep(x,EdgeID,EdgeAttr,batch)
        GraphX = global_mean_pool(NodeX, batch)
        return GraphX
    
    def Predict(self,GraphX):
        pred = self.Linear1(GraphX)
        pred = self.relu(pred)
        pred = self.Linear2(pred)
        self.readout = pred
        
        # un-normalized probability.
        return pred
    
    def ResetParam(self):
        with torch.no_grad():
            for p in self.parameters():
                p.unique(-1.0,1.0)

# In[Settings]
if __name__ == "__main__":
    '''Random seed.'''
    seed = 0
    GlobalSeed(seed)
    
    Args = Args()
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(Args.cuda))
    else:
        device = torch.device('cpu')
    
# In[Dataset]
    Train = ba3motif(Args.datapath,mode = 'train')
    Valid = ba3motif(Args.datapath,mode = 'valid')
    Test = ba3motif(Args.datapath,mode = 'test')
    
    TrainLoader = DataLoader(Train,
                             batch_size = Args.batchsize,
                             shuffle = True)
    ValidLoader = DataLoader(Valid,
                             batch_size = Args.batchsize,
                             shuffle = True)
    TestLoader = DataLoader(Test,
                            batch_size = 1,
                            shuffle = False)
    # In[Model]
    GNN = BA3Net(Args.numlayers).to(device)
    '''opt for optimizer, sdl for scheduler'''
    opt = torch.optim.Adam(GNN.parameters(),lr = Args.lr)
    sdl = ReduceLROnPlateau(opt, mode = 'min', factor = 0.8, 
                            patience = 10, min_lr = 1e-4)
    
    MinLoss = None

# In[Train & Test]
    for epoch in range(1,Args.epoch + 1):
        t1 = time.time()
        
        
        Loss,Acc = TrainProxy(TrainLoader,GNN,opt,device,CrossEntropyLoss())
        ValLoss,ValAcc = TestProxy(ValidLoader,GNN,device,CrossEntropyLoss())
        if MinLoss is None or ValLoss < MinLoss:
            MinLoss = ValLoss
        sdl.step(ValLoss)
        t2 = time.time()
        
        lr = sdl.optimizer.param_groups[0]['lr']
        if epoch % Args.testepoch == 0:
            TestLoss,TestAcc = TestProxy(TestLoader,GNN,device,CrossEntropyLoss())
            t3 = time.time()
            print('Epoch{:4d}[{:.3f}]: LR:{:.5f},'.format(epoch,t3-t1,lr),
                  'Train Loss: {:.5f}, Train Acc: {:.5f}'.format(Loss,Acc),
                  'Valid Loss: {:.5f}, Valid Acc: {:.5f}'.format(ValLoss,ValAcc),
                  'Test Loss: {:.5f}, Test Acc: {:.5f}'.format(TestLoss,TestAcc))
        else:
            print('Epoch{:4d}[{:.3f}]: LR:{:.5f},'.format(epoch,t2-t1,lr),
                  'Train Loss: {:.5f}, Train Acc: {:.5f}'.format(Loss,Acc),
                  'Valid Loss: {:.5f}, Valid Acc: {:.5f}'.format(ValLoss,ValAcc))

# In[Save]
    SavePath = 'ba3net.pt'
    if not op.exists(Args.modelpath):
        os.makedirs(Args.modelpath)
    torch.save(GNN.cpu(),op.join(Args.modelpath,SavePath))
