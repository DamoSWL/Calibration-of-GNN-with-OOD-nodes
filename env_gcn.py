import os.path as osp
import numpy as np
import random
import os
from sys import argv
import torch
from scipy.sparse import csr_matrix
from collections import defaultdict
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
from time import time
import scipy.sparse as sp
from utils.util import  data_loader_OOD, sparse_mx_to_torch_sparse_tensor,manual_seed,accuracy
from utils.normalization import fetch_normalization
import math
import torch.optim as optim
from collections import defaultdict
import matplotlib.pyplot as plt


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
  
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
  

    def init_model(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)



class ECELoss(nn.Module):
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.exp(logits)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece +=  torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class gcn_env_gcn(object):
    def __init__(self,dataset='cora',datapath='data',task_type='semi',args=None):
        
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')       
        self.dataset = dataset

        self.adj, _, self.features,_,self.labels,self.idx_train,self.idx_val, \
        self.idx_test,self.degree,self.learning_type,self.idx_test_ood, \
        self.idx_test_id,self.test_mask_id,self.test_mask_ood = data_loader_OOD(dataset, datapath, "NoNorm", False, task_type)

        self.model_path = args.model_path
        # weighted_adj= self.weighted_adj(1.0)
        # self.adj = sp.coo_matrix(weighted_adj)

        self.adj = self.adj.astype('float64')
          
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)
        self.idx_train = torch.LongTensor(self.idx_train)
        self.idx_val = torch.LongTensor(self.idx_val)
        self.idx_test = torch.LongTensor(self.idx_test)
        self.idx_test_ood = torch.LongTensor(self.idx_test_ood) 
        self.idx_test_id = torch.LongTensor(self.idx_test_id) 
        self.test_mask_id = torch.BoolTensor(self.test_mask_id) 
        self.test_mask_ood = torch.BoolTensor(self.test_mask_ood)

        self.features = self.features.to(self.device)
        self.labels = self.labels.to(self.device)
        self.idx_train = self.idx_train.to(self.device)
        self.idx_val= self.idx_val.to(self.device)
        self.idx_test = self.idx_test.to(self.device)
        self.idx_test_ood = self.idx_test_ood.to(self.device)
        self.idx_test_id = self.idx_test_id.to(self.device)
        self.test_mask_id =self.test_mask_id.to(self.device)
        self.test_mask_ood = self.test_mask_ood.to(self.device)

        self.nfeat = self.features.shape[1]
        self.ndata = self.features.shape[0]
        self.nclass = int(self.labels.max().item() + 1)

        self.ECEFunc = ECELoss()

        self.candidate_adj_hop = 2  # variable used to sample the edges from the centered node. 
        self.validate_node_hop = 1  # varibale used to sample the nodes for the calculation of the reward signal

        self.candidate_adj = None   # the container of the sampled edges for the adjustment of the weights
        self.candidate_adj_num = 1000
        self.node_map = defaultdict(list)  # the container used to obtain the corresponding nodes for the calculation of the reward signals

        self.node_1 = None  # the source nodes of the sampled edges
        self.node_2 = None  # the target nodes of the sampled edges

        self.batch_size = 32
        self.edge_pointer = 0 # the indicator to the current position of the traversed edges
        self.cnt = 0

        self.total_timesteps = math.ceil(self.candidate_adj_num * 1.0 /self.batch_size)

        self.policy = None

        self.args = args

        self.model = GCN(nfeat=self.nfeat, nclass=self.nclass, nhid=self.args.hidden,dropout=0.5)
        self.model = self.model.to(self.device)
        self.gcn_optimizer = optim.Adam(self.model.parameters(), lr=1e-2,weight_decay=5e-4)

        self.weight = args.weight
        self.seed = args.seed


    def init_model(self):
        self.model.init_model()
        self.model = self.model.to(self.device)



    def init_candidate_adj(self):
        node_num = 10
        target_nodes = torch.cat((self.idx_train,self.idx_val)).cpu().numpy()
        target_nodes = np.random.permutation(target_nodes)[:node_num]
        logging.info(f'candidate adj')
        self.candidate_adj = self.get_candidate_adj(target_nodes)
        logging.info(f'node map')
        self.get_val_node_map()
    
   

    def get_candidate_adj(self,idx):
        candidate_adj = []
        for node in idx:
            if len(candidate_adj) >= self.candidate_adj_num:
                break
            self.get_candidate_adj_from_single_node(candidate_adj,node,self.candidate_adj_hop,None)
           
        
        return np.array(candidate_adj).T


    def get_candidate_adj_from_single_node(self,candidate_adj,node,hop,pre_node):

        self.adj = sp.coo_matrix(self.adj)
        edge_index = np.vstack((self.adj.row,self.adj.col)) # [2.N]
        if hop > 0:
            mask = edge_index[0] == node
            neighbor_nodes = edge_index[1,mask]
            if len(candidate_adj) <= self.candidate_adj_num:
                candidate_adj.extend([(node,adj_node) for adj_node in neighbor_nodes if adj_node != pre_node])
            
            for adj_node in neighbor_nodes:
                self.get_candidate_adj_from_single_node(candidate_adj,adj_node.item(),hop-1,node)



    def get_val_node_map(self):
        nodes = self.candidate_adj.flatten().tolist()
        nodes = set(nodes)
        for node in nodes:
            if len(self.node_map[node]) == 0:
                adj_nodes = self.get_val_nodes(node,self.validate_node_hop)
                self.node_map[node].extend(adj_nodes)
    


    def get_val_nodes(self,node,hop):
        idx = torch.cat((self.idx_train,self.idx_val)).cpu().numpy()
        self.adj = sp.coo_matrix(self.adj)
        edge_index = np.vstack((self.adj.row,self.adj.col))
       
        val_nodes = []
        if hop >= 0:
            if node in idx:
                val_nodes.append(node)

            mask = edge_index[0] == node
            neighbor_nodes = edge_index[1,mask]

            for adj_node in neighbor_nodes:
                if adj_node in idx:
                    val_nodes.append(adj_node)


        return val_nodes



    def update_adj(self):  # update all the edge weights in a graph from the output of the Q network
        cur = 0
        batch = 5000
        v = []

        self.adj = sp.coo_matrix(self.adj)
        edge_index = np.vstack((self.adj.row,self.adj.col))

        self.adj = sp.csr_matrix(self.adj)


        while cur < edge_index.shape[1]:
            node_1 = edge_index[0,cur:cur+batch]
            node_2 = edge_index[1,cur:cur+batch]
            state = self.features[node_1] + self.features[node_2]
            state /= 2.0
            cur += batch      
            v.extend(self.policy.step_adj_weight(state))
            

        v = np.array(v)

        a = v[v < 1.0]
        logging.info(f'adjust {a.shape[0]*1.0/v.shape[0]}')

        self.adj[edge_index[0,:],edge_index[1:]] = v

        self.adj = sp.coo_matrix(self.adj)

        os.makedirs('adj',exist_ok=True)
        np.savez(f'adj/{self.dataset}_adj',value = self.adj.data, row=self.adj.row,col=self.adj.col,shape=self.adj.shape)


    def reset(self):

        self.node_1 = self.candidate_adj[0,:self.batch_size]
        self.node_2 = self.candidate_adj[1,:self.batch_size]

        state = self.features[self.node_1] + self.features[self.node_2]
        state /= 2.0

        self.cnt = 0
        self.edge_pointer = 0

        return state


    def validate_nodes(self,adj):  # obtain the reward signal
        val_acc = []
   
        self.model.eval()

        adj = sp.csr_matrix(adj)
        adj = self.preprocess_adj('AugNormAdj',adj)   

        with torch.no_grad():
            output = self.model(self.features, adj)
           
        
        for i in range(self.node_1.shape[0]):
            val_nodes = []
            mask = np.zeros(self.ndata).astype(np.bool_)

            node = self.node_1[i]
            val_nodes.extend(self.node_map[node])
            
            node = self.node_2[i]
            val_nodes.extend(self.node_map[node])

            assert len(val_nodes) > 0  
            mask[val_nodes] = True
            acc = accuracy(output[mask], self.labels[mask])
            val_acc.append(acc)

        val_acc = torch.as_tensor(val_acc)

        return val_acc
   

    def step(self, actions):  # obtain the reward signal from the action (adjusting the edge weights) and the next batch of edges

        v = np.ones(actions.shape[0])
        mask = actions.cpu().numpy()<1    
        v[mask] = self.weight

        self.adj = sp.csr_matrix(self.adj)
        self.adj[self.node_1,self.node_2] = v

        reward = self.validate_nodes(self.adj)

        self.edge_pointer += self.batch_size
        self.node_1 = self.candidate_adj[0,self.edge_pointer:self.edge_pointer+self.batch_size]
        self.node_2 = self.candidate_adj[1,self.edge_pointer:self.edge_pointer+self.batch_size]


        next_state = self.features[self.node_1] + self.features[self.node_2]
        next_state /= 2.0

        self.cnt += 1
        if self.cnt >= self.total_timesteps:
            done = 1
        else:
            done = 0
        
        done = [done] * self.batch_size
        done = torch.as_tensor(done,dtype=torch.bool)
       
        return next_state, reward, done



    def preprocess_adj(self, normalization, adj):
        adj_normalizer = fetch_normalization(normalization)
        r_adj = adj_normalizer(adj)
        r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()  
        r_adj = r_adj.to(self.device)
        return r_adj



    def train(self,adj,total_epoch,save=True):

        self.model.train()

        adj = sp.csr_matrix(adj)
        adj = self.preprocess_adj('AugNormAdj',adj)

        for epoch in range(total_epoch):
            self.model.train()
            output = self.model(self.features,adj)
            loss = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
            loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
          

            self.gcn_optimizer.zero_grad()
            loss.backward()
            self.gcn_optimizer.step()

            train_acc = accuracy(output[self.idx_train].detach(), self.labels[self.idx_train].detach()) 
            val_acc = accuracy(output[self.idx_val].detach(), self.labels[self.idx_val].detach())

            if save:
                path = Path(self.model_path)
                if not path.exists():
                    path.mkdir()
                torch.save({'model_state_dict': self.model.state_dict()},str(path / 'GCN_best_model.pth'))
      



    
    def validate(self,adj):
        self.model.eval()

        path = Path(self.model_path)
        checkpoint = torch.load(str(path / 'GCN_best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        adj = sp.csr_matrix(adj)
        adj = self.preprocess_adj('AugNormAdj',adj)
    
        with torch.no_grad():
            output = self.model(self.features, adj)
            loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        
        acc = accuracy(output[self.idx_val], self.labels[self.idx_val])
        ECE = self.ECEFunc(output[self.idx_val], self.labels[self.idx_val])


        return acc.item(), ECE.item(),loss_val.item()




    def test(self,adj):

        average_acc = 0
        average_ece = 0
        
        adj = sp.csr_matrix(adj)
        adj = self.preprocess_adj('AugNormAdj',adj)

        for i in range(10): 
            test_idx_i = np.random.choice(self.idx_test.cpu().numpy(), size=1000, replace=False)
            test_idx_filtered = [ele for ele in test_idx_i if ele in self.idx_test_id.cpu().numpy()]
            test_idx_filtered = torch.tensor(test_idx_filtered)
            test_idx_filtered = test_idx_filtered.to(self.device)
            acc, ece = self.test_single(adj,test_idx_filtered,i)
            average_acc += acc.item()
            average_ece += ece.item()

        return average_acc/10.0, average_ece/10.0



    def test_single(self,adj,idx,i):
        self.model.eval()

        path =Path(self.model_path)
        checkpoint = torch.load(str(path / 'GCN_best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])


        with torch.no_grad():
            output = self.model(self.features, adj)
        

        loss = F.nll_loss(output[idx], self.labels[idx])
        acc = accuracy(output[idx], self.labels[idx])
        ECE = self.ECEFunc(output[idx], self.labels[idx])


        return acc, ECE

 

                