'''
Base explainer.
'''
# In[Import]
import os
import math
import numpy as np

import torch

# from utils.math import Harmonic
# from gnn import *

# In[Explainer]
class Explainer(object):
    def __init__(self,device,ModelPath):
        # super().__init__()
        self.device = device
        self.ModelPath = ModelPath
        self.model = torch.load(self.ModelPath).to(self.device)
        self.model.eval()
        
        self.ModelName = self.model.__class__.__name__
        self.name = self.__class__.__name__
        
        self.result = None
        # self.VisDict = None
        
    def __relabel__(self,g,pos_edge):
        Nodes = torch.unique(pos_edge)
        X = g.x[Nodes]
        Batch = g.batch[Nodes]
        
        Row, _ = pos_edge
        NodeId = Row.new_full((g.num_nodes,), -1)
        NodeId[Nodes] = torch.arange(Nodes.size(0),device = self.device)
        EdgeId = NodeId[pos_edge]
            
        return X, EdgeId, Batch
    
    @staticmethod
    def Norm(score):
        score[score < 0] = 1e-6
        return score / score.sum()
        
    def explain(self,g,**kargs):
        '''
        Parameters
        ----------
        g : object
            Graph to explained.

        Raises
        ------
        NotImplementedError
            Virtual function to be implemented in child class.

        Returns
        -------
        Edge Score : Tensor
            Score for each edge.

        '''
        raise NotImplementedError
        
    def get_subgraph_(self, result = None, relabel = True):
        assert result or self.result is not None
        if result is None:
            g, exp = self.result
        else:
            g, exp = result
            
        sub_g = g.clone()
        sub_g.y = g.y
        
        ratio = 0.7
        TopK = int(np.round((ratio * g.num_edges)))
        TopK = min(max(TopK, 1), g.num_edges)
        pos_edge_id = np.argsort(-exp)[:TopK]
        
        pos_edge = g.edge_index[:,pos_edge_id]
        sub_g.edge_attr = g.edge_attr[pos_edge_id,:]
        
        if relabel:
            sub_g.x, sub_g.edge_index, sub_g.batch = \
                self.__relabel__(g, pos_edge)

        return sub_g
    
    def del_subgraph_(self, result = None, relabel = True):
        assert result or self.result is not None
        if result is None:
            g, exp = self.result
        else:
            g, exp = result
        
        del_g = g.clone()
        del_g.y = g.y
        
        ratio = 0.70
        DelK = int(np.round(ratio * g.num_edges))
        DelK = min(max(DelK, 1), g.num_edges)
        DelK = g.num_edges - DelK
        del_edge_id = np.argsort(exp)[:DelK]
        
        del_edge = g.edge_index[:,del_edge_id]
        del_g.edge_attr = g.edge_attr[del_edge_id,:]
        
        if relabel:
            del_g.x, del_g.edge_index, del_g.batch = \
                self.__relabel__(g, del_edge)
        return del_g

    def F1Score(self,
                RatioList = [0.1 * i for i in range(1,11)]):
        '''
        Parameters
        ----------
        RatioList : int, optional
            List of explanation / true explanation

        Returns
        -------
        recall : float.
            Recall of explaination.
            Recall = TP / (TP + FN).
        precison : float.
            Precision of explanation.
            Precision = TP / (TP + FP).
        F1 : float.
            F1 Score of explanation.
            F1 = 2*Re*Pr / (Re + Pr).
        '''
        assert self.result is not None
        g, exp = self.result
        '''
        GTMask for ground truth mask.
        In: g.GTMask
        Out: [array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 
                     1., 1., 1., 1., 1.,1., 1., 1., 1., 1., 1., 1., 
                     1., 1., 1., 1., 1., 1., 1.])]
            <class 'numpy.ndarray'> 
        '''
        TRUE = g.GTMask[0]
        LENGTH = TRUE.size
        
        Recall, Precision = [],[]
        
        for ratio in RatioList:
            TopK = np.round(ratio * LENGTH)
            TopK = int(TopK)
            index = np.argsort(-exp)[:TopK]
            equal = TRUE[index]
            
            recall = float(equal.sum()) / float(TRUE.sum())
            precision = float(equal.sum()) / TopK
            
            Recall.append(recall)
            Precision.append(precision)
            
        F1 = [Harmonic(a,b) for a,b in zip(Recall,Precision)]
            
        return Recall, Precision, F1

    def Fidelity(self):
        '''
        Returns the fidelity of explanation.
        '''
        assert self.result is not None
        g, exp = self.result
        
        # Fidelity = []
        
        '''Prediction of full graph.'''
        self.model(g)
        yhat = self.model.readout.argmax(dim = 1)
        
        exp_sub = self.get_subgraph_()
        self.model(exp_sub)

        exp_hat = self.model.readout.argmax(dim = 1)
        fid = exp_hat == yhat
        fid = fid.item()

        return fid
    
    def FnsScore(self):
        assert self.result is not None
        g, exp = self.result
        
        '''Prediction of full graph.'''
        self.model(g)
        yhat = self.model.readout.argmax(dim = 1)
        
        ps = self.Fidelity()
        
        del_sub = self.del_subgraph_()
        self.model(del_sub)

        
        del_hat = self.model.readout.argmax(dim = 1)
        pn = del_hat != yhat
        pn = pn.item()
        
        return ps, pn
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            