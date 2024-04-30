from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

# from utils import load_data, accuracy
from utils import  accuracy,manual_seed,plot_acc_calibration
from data import load_data_ood
from models import GCN
from util_ece import *
from util_log import *
import os




# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=80, help='Random seed.')
parser.add_argument('--epochs', type=int, default=600,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    help='dataset name')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


manual_seed(args.seed)


set_logger('acc','acc.txt')
set_logger('ece','ece.txt')


# Load data
adj, features, labels, idx_train, idx_val, idx_test_id, idx_test = load_data_ood(args.dataset)



# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)



optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=args.weight_decay)


if args.cuda:
    model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test_id = idx_test_id.to(device)
    idx_test = idx_test.to(device)



ECEFunc = ECELoss()

best_loss = 100
bad_counter = 0
patience = 100

 

def train(epoch):
    global best_loss
    global bad_counter

    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    print(f'output {output.shape}')

    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    if loss_val <  best_loss:
        best_loss = loss_val
        torch.save({'model_state_dict': model.state_dict()}, 'GCN_best_model.pth')
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter >= patience:
        return False

    return True

def test(i,idx_test_sub):
    checkpoint = torch.load('GCN_best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    output = model(features, adj)


    loss_test = F.nll_loss(output[idx_test_sub], labels[idx_test_sub])
    acc_test = accuracy(output[idx_test_sub], labels[idx_test_sub])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    nclass = labels.max().item() + 1
    

    ECE = ECEFunc(output[idx_test_sub], labels[idx_test_sub])

    # if i == 0:
    #     plot_acc_calibration(idx_test_sub, output, labels,args.seed,args.dataset, n_bins=10, title=None)
    return acc_test.item(), ECE.item()



# Train model
t_total = time.time()
for epoch in range(args.epochs):
    if not train(epoch):
        break
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# rng = np.random.default_rng(args.seed)
ACC = []
ECE = []
for i in range(10):
    idx_test_sub = np.random.choice(idx_test.cpu().numpy(), size=1000, replace=False)
    index = labels[idx_test_sub.tolist()]>=0
    index = index.cpu().numpy().astype(np.bool_)
    idx_test_sub = idx_test_sub[index]
    idx_test_sub = torch.LongTensor(idx_test_sub)
    idx_test_sub = idx_test_sub.to(device)
    acc ,ece = test(i,idx_test_sub)
    ACC.append(acc)
    ECE.append(ece)

ACC = np.array(ACC)
ECE = np.array(ECE)

ACC = ACC.mean()
ECE = ECE.mean()

logging.getLogger('acc').info(ACC)
logging.getLogger('ece').info(ECE)

