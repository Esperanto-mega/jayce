# In[Import]
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from base import Explainer
# from explainer.common import *
# from explainer.overload import *
from vae_v2 import VAE, AlphaNet
# from gnn.vae import AlphaNet

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import add_self_loops

# In[V-InfoR]
class vinfor(Explainer):
    # Parameter for loss function.
    eps = {'stable': 1e-6,
           'size': 5e-3,
           'entropy': 1}
    
    # Clamp probability.
    prob = {'min': 0.01,
            'max': 0.99}
    
    def __init__(self, device, target_model_path, node_dim, edge_dim, vae_hidden, vae_out, alpha_hidden, Classes = 3):
        super().__init__(device, target_model_path)
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.vae_hidden = node_dim
        self.vae_out = node_dim
        self.alpha_hidden = alpha_hidden
        self.device = device

        self.vae = VAE(self.node_dim, self.vae_hidden, self.vae_out, device = self.device).to(self.device)

        self.alpha_net = AlphaNet(self.edge_dim, self.vae_out, self.alpha_hidden, out_dim = 1).to(self.device)
        
        '''
        # In: node-feature dim
        # Out: hidden dim
        node_dim = NodeChannels
        self.Vae = VAE(node_dim, vae_hidden, vae_out, device).to(device)
        
        # 2 * Hidden: edge(u,v) embedding 
        # from node embedding u,v
        edge_dim = EdgeChannels
        self.AlphaNet = AlphaNet(edge_dim, vae_out, alpha_hidden, out_dim = 1).to(device)
        # self.Encoder = SubgraphNet(NodeChannels,EdgeChannels,
        #                            Hidden,Layers).to(device)
        '''
    
    def __set_mask__(self,edge_mask,model):
        for module in model.modules():
            if isinstance(module,MessagePassing):
                module._explain = True
                module._edge_mask = edge_mask
    
    def __clear_mask__(self,model):
        for module in model.modules():
            if isinstance(module,MessagePassing):
                module._explain = False
                module._edge_mask = None
    
    def get_adj_pred_and_z(self, triple):
        Adj_pred = ()
        Sampled_z = ()
        for g in triple:
            # adj_pred = (self.Vae(g,g.edge_attr),)
            # print('get_adj_pred num_nodes:',g.num_nodes)
            # print('name:',g.name)
            
            adj_pred = (self.Vae(g),)
            sampled_z = (self.Vae.sampled_z,)
            
            Adj_pred = Adj_pred + adj_pred
            Sampled_z = Sampled_z + sampled_z
        return Adj_pred, Sampled_z

    def get_alpha(self,triple):
        Alpha = ()
        for g in triple:
            alpha = (self.AlphaNet(g).view(-1),)
            Alpha = Alpha + alpha
            
        return Alpha
    
    def probability_represent(self, Alpha, tau = 0.3, training = True):
        pM = ()
        for alpha in Alpha:
            if training:
                noise = torch.rand(alpha.size()).to(self.device)
                pm = torch.log2(noise) - torch.log2(1.0 - noise)
                pm = (pm + alpha) / tau + self.eps['stable']
                pm = torch.clamp(pm.sigmoid(),
                                 self.prob['min'], self.prob['max'])
            else:
                pm = torch.clamp(alpha.sigmoid(),
                                 self.prob['min'], self.prob['max'])
            pm = (pm,)
            pM = pM + pm
            
        return pM
    
    def get_yhat(self, triple):
        Yhat = ()
        for g in triple:
            # print('x:',g.x.shape)
            # print('edge_index:',g.edge_index.shape)
            # print('edge_attr',g.edge_attr.shape)
            yhat = (self.model(g),)
            Yhat = Yhat + yhat
            
        return Yhat
    
    def get_sub_yhat(self,triple,pM):
        Sub_Yhat = ()
        for (g,pm) in zip(triple,pM):
            # print('pm shape:',pm.shape)
            # print('edge_index shape:',g.edge_index.shape)
            self.__set_mask__(pm,self.model)
            sub_yhat = (self.model(g),)
            self.__clear_mask__(self.model)
            Sub_Yhat += sub_yhat
            
        return Sub_Yhat
    
    def get_graph_rep(self, triple, pM):
        Graph_rep = ()
        for (g,pm) in zip(triple,pM):
            self.__set_mask__(pm,self.model)
            nodes_index = torch.unique(g.edge_index)
            x = g.x[nodes_index]
            batch = g.batch[nodes_index]
            graph_rep = (global_mean_pool(x, batch),)
            self.__clear_mask__(self.model)
            
            Graph_rep = Graph_rep + graph_rep
        
        return Graph_rep
    
    def edge_index_2_adj_matrix(self,true_triple):
        Adj_matrix = ()
        for g in true_triple:
            # edge_index = remove_self_loops(g.edge_index)[0]
            adj_matrix = to_dense_adj(g.edge_index)
            adj_matrix = torch.squeeze(adj_matrix, dim = 0)
            adj_matrix = torch.triu(adj_matrix,diagonal = 1)
            # print('adj shape:',adj_matrix.shape[0])
            # print('edge_index_2 num nodes:',g.num_nodes)
            # print('name:',g.name)
            
            # For outliers
            # if adj_matrix.shape[0] != g.num_nodes:
            #     # print('adj shape:',adj_matrix.shape[0])
            #     # print('num nodes:',g.num_nodes)
            #     pad = nn.ZeroPad2d((0, g.num_nodes - adj_matrix.shape[0],
            #                         0, g.num_nodes - adj_matrix.shape[0]))
            #     adj_matrix = pad(adj_matrix)
            
            adj_matrix = (adj_matrix,)
            
            Adj_matrix += adj_matrix
            
        return Adj_matrix
    
    def adj_matrix_2_edge_index(self,adj_triple):
        Edge_index = ()
        Edge_weight = ()
        for adj in adj_triple:
            # edge_index = (dense_to_sparse(adj)[0],)
            edge_index, edge_weight = dense_to_sparse(adj)
            edge_index = (edge_index,)
            edge_weight = (torch.unsqueeze(edge_weight, dim = 1),)
            Edge_index += edge_index
            Edge_weight += edge_weight
        
        return Edge_index, Edge_weight
    
    def VAELoss(self, adj_pred, adj):
        # print('adj-pred:',adj_pred)
        # print('adj:',adj)
        similar_loss = F.binary_cross_entropy(adj_pred, adj)
        
        gauss_term = 1 + 2 * self.Vae.log_stddev - self.Vae.mean ** 2
        gauss_term = gauss_term - torch.exp(self.Vae.log_stddev) ** 2
        gauss_term = gauss_term.sum(1).mean()
        kl_loss = 0.5 / adj_pred.size(0) * gauss_term

        Loss = similar_loss + kl_loss
        return Loss
    
    def FidelityLoss(self,yHat,Sub_yHat,pM):
        # Normalized probability.
        yHat = yHat.softmax(dim = 1)
        CELoss = nn.CrossEntropyLoss()
        FidLoss = CELoss(Sub_yHat, yHat)
        
        # SizeLoss = torch.sum(pM) * self.eps['size']
        
        pi = 0.5
        EntropyLoss = - (pM * torch.log((pM + 1e-6) / pi) - \
            (1 - pM) * torch.log((1 - pM + 1e-6)) / (1 - pi)) 
        EntropyLoss = torch.sum(EntropyLoss) * self.eps['entropy']
        
        # Loss = FidLoss + SizeLoss
        Loss = FidLoss + EntropyLoss
        # Loss = FidLoss + SizeLoss + EntropyLoss
        
        return Loss
    
    def ContrastLoss(self, vector):
        for v in vector:
            v = v / v.norm(dim = 1, keepdim = True)
            
        P = torch.mm(vector[0],vector[2].T)
        N = torch.mm(vector[0],vector[1].T)
        
        N = torch.log(torch.exp(P) + torch.exp(N) + self.eps['stable'])
        
        Loss = (N - P).view(-1)
        return Loss
    
    def vae_train(self,g):
        # print('num_nodes:',g.x.shape[0])
        Adj_pred = self.Vae(g)
        # edge_index = remove_self_loops(g.edge_index)[0]
        # print('non-outliers:',torch.unique(edge_index).shape[0])
        Adj = to_dense_adj(g.edge_index)
        Adj = torch.squeeze(Adj,dim = 0)
        Adj = torch.triu(Adj,diagonal = 1)
        # print('Adj_pred:',Adj_pred)
        # print('Adj:',Adj)
        VaeLoss = self.VAELoss(Adj_pred, Adj)
        # print('VaeLoss:',VaeLoss)
        return VaeLoss
    
    def exp_train(self,triple):
        x = []
        for g in triple:
            x.append(g.x.detach())
            print('feature:',g.x.shape)
            # feature: mut-14
            g.x = self.model.NodeRep(g.x,g.edge_index,g.edge_attr,g.batch)
            print('node embedding:',g.x.shape)
            # node embedding: mut-32
        Alpha = self.get_alpha(triple)
        pM = self.probability_represent(Alpha)
        
        # -------- Contrastive Loss ----------- #
        # ------------------------------------- #
        Graph_rep = self.get_graph_rep(triple)
        CtsLoss = self.ContrastLoss(Graph_rep)
        # ------------------------------------- #
        
        for i in range(len(triple)):
            triple[i].x = x[i]
        # ------------- Predition ------------- #
        # ------------------------------------- #
        yHat = self.get_yhat(triple)
        Sub_yHat = self.get_sub_yhat(triple,pM)
        # ------------------------------------- #
        
        
        # ------------------ Fidelity Loss ------------------ #
        # --------------------------------------------------- #
        FidLoss = 0
        for (rec, sub, pm) in zip(yHat, Sub_yHat, pM):
            FidLoss += self.FidelityLoss(rec, sub, pm)
        # --------------------------------------------------- #
        
        return CtsLoss, FidLoss
    
    def pretrain(self, triple):
        Adj_pred, Sampled_z = self.get_adj_pred_and_z(triple)
        Adj = self.edge_index_2_adj_matrix(triple)
        
        VaeLoss = 0
        x_raw = []
        for i in range(len(Adj_pred)):
            VaeLoss += self.VAELoss(Adj_pred[i],Adj[i])
            x_raw.append(triple[i].x.detach())
            triple[i].x = Sampled_z[i]

        Alpha = self.get_alpha(triple)
        pM = self.probability_represent(Alpha)
        
        for i in range(len(x_raw)):
            triple[i].x = x_raw[i]
        
        FidLoss = 0
        yhat = self.get_yhat(triple)
        exp_yhat = self.get_sub_yhat(triple,pM)
        for i in range(len(yhat)):
            FidLoss += self.FidelityLoss(yhat[i],exp_yhat[i],pM[i])
        return VaeLoss, FidLoss

    def exp_explain(self, g, finetune = True,
                    ft_lr = 1e-5, ft_epoch = 20, ft_min_loss = None,
                    visualize = False, vis_ratio = 0.4, savefig = False):
        if not finetune:
            x = g.x.detach()
            g.x = self.model.NodeRep(g.x,g.edge_index,g.edge_attr,g.batch)
            
            unitg = (g,)
            alpha = self.get_alpha(unitg)
            pm = self.probability_represent(alpha)[0]
            
            g.x = x
            explain = pm.detach().cpu().numpy()
            self.result = (g,explain)
            
            if visualize:
                self.visualize(g,explain,vis_ratio,savefig)
            return explain
        
        FineNet = copy.deepcopy(self.AlphaNet)
        parameters = FineNet.parameters()
        opt = torch.optim.Adam(parameters,ft_lr)
        yhat = self.model(g)
        
        for _ in range(ft_epoch):
            x = g.x.detach()
            g.x = self.model.NodeRep(g.x,g.edge_index,g.edge_attr,g.batch)
            alpha = (FineNet(g),)
            pm = self.probability_represent(alpha)[0]
            
            g.x = x
            sub_yhat = self.model(g,edge_weight = pm)
            
            FidLoss = self.FidelityLoss(yhat, sub_yhat, pm)
            opt.zero_grad()
            FidLoss.backward(retain_graph = True)
            opt.step()
            
        explain = pm.detach().cpu().numpy()
        self.result = (g,explain)
        
        if visualize:
            self.visualize(g,explain,vis_ratio,savefig)
        return explain

    def explain(self,g,finetune = False,
                ft_lr = 1e-5, ft_epoch = 10,
                visualize = False, vis_ratio = 0.4, savefig = False):
        tuple_g = (g,)
        if not finetune:
            Sampled_z = self.get_adj_pred_and_z(tuple_g)[1]
            x_raw = g.x.detach()
            g.x = Sampled_z[0]
            Alpha = self.get_alpha(tuple_g)
            pM = self.probability_represent(Alpha)
            g.x = x_raw
            
            explain = pM[0].detach().cpu().numpy()
            self.result = (g,explain)
            
            if visualize:
                self.visualize(g,explain,vis_ratio,savefig)
            return explain
        
        # Fine-tune
        FineNet = copy.deepcopy(self.AlphaNet)
        params = FineNet.parameters()
        opt = torch.optim.Adam(params,ft_lr)
        
        for _ in range(ft_epoch):
            Sampled_z = self.get_adj_pred_and_z(tuple_g)[1]
            x_raw = g.x.detach()
            g.x = Sampled_z[0]
            Alpha = self.get_alpha(tuple_g)
            pM = self.probability_represent(Alpha)
            g.x = x_raw
            
            yhat = self.get_yhat(tuple_g)[0]
            exp_yhat = self.get_sub_yhat(tuple_g,pM)[0]
            FidLoss = self.FidelityLoss(yhat,exp_yhat,pM[0])
            
            opt.zero_grad()
            FidLoss.backward()
            opt.step()
        
        explain = pM[0].detach().cpu().numpy()
        if visualize:
            self.visualize(g,explain,vis_ratio,savefig)
        self.result = (g,explain)
        return explain

    def Remap(self,device):
        self.device = device
        self.model = self.model.to(device)
        self.Encoder = self.Encoder.to(device)