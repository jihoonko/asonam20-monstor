import argparse
import sys
import os

import torch
import numpy as np
from utils import load_data
from models import MONSTOR
import dgl
import time
from scipy.stats import spearmanr, pearsonr

torch.set_num_threads(4)
graph_names = ['Extended', 'Celebrity', 'WannaCry']
prob_names = ['BT', 'JI', 'LP']

def IE():
    global graph_names
    
    # use only single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    test_labels = [(graph_name, 'test', args.prob) for graph_name in graph_names]
    print('test_labels: {}'.format(test_labels))
    
    g, sX, sy, X, y, _g, _sX, _sy = load_data(train_labels=[], val_labels=[], test_labels=test_labels)
    model = MONSTOR(in_feats=args.input_dim, n_hidden=args.hidden_dim, n_layers=args.layer_num).cuda()
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    
    # for fixed match size
    batch_size = 20
    bg = {}
    for lstr, graph in _g.items():
        bg[lstr] = dgl.batch([graph for _ in range(batch_size)])
    
    # hyperparameters
    n_features = args.input_dim
    n_stacks = args.n_stacks
    
    with torch.no_grad():
        for tl in test_labels:
            lstr = '_'.join(tl)
            V = _g[lstr].number_of_nodes()
            x = torch.zeros(bg[lstr].number_of_nodes(), n_features + n_stacks, requires_grad=False).cuda()
            gt, preds = [], [] # gt: ground truth influence, preds: predicted influence
            for idx in range(0, _sX[lstr].shape[0], batch_size):
                gt = gt + [torch.sum(_sy[lstr][idx+i]).item() for i in range(batch_size)]
                x[:, n_features-2] = x[:, n_features-1] = torch.t(_sX[lstr][idx:idx+batch_size].view(-1).cuda())
                for j in range(n_features, n_features + n_stacks):
                    x[:, j] = model(bg[lstr], x[:, j-n_features:j]).data
                    x[:, j-1] = x[:, j] - x[:, j-1]
                preds = preds + [torch.sum(x[:, (n_features + n_stacks - 1)][i*V:(i+1)*V]).item() for i in range(batch_size)]
            
            # Compute Spearman Rank Correlation and Pearson Correlation
            scorr, _ = spearmanr(gt[:], preds[:])
            pcorr, _ = pearsonr(gt[:], preds[:])
            
            print("for target graph {} | Spearmanr: {}, Pearsonr: {}".format(tl[0], scorr, pcorr))
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, help="input dimension")
    parser.add_argument("--hidden-dim", type=int, help="hidden dimension")
    parser.add_argument("--layer-num", type=int, help="# of layers")
    parser.add_argument("--checkpoint-path", help="path of target checkpoint")
    parser.add_argument("--prob", help="target activation probablity")
    parser.add_argument("--n-stacks", type=int, help="number of stacks")
    parser.add_argument("--gpu", type=int, help="gpu number")
    
    args = parser.parse_args()
    if not args.input_dim: args.input_dim = 4
    
    # argument validation
    if type(args.input_dim) != int or args.input_dim < 2:
        print("invalid input dimension")
        sys.exit()
    if type(args.hidden_dim) != int:
        print("invalid hidden dimension")
        sys.exit()
    if args.gpu is None or args.gpu < 0 or args.gpu > 3:
        print("invalid gpu id")
        sys.exit()
    if args.layer_num is None or args.layer_num < 1:
        print("invalid layer number")
        sys.exit()
    if args.checkpoint_path is None or (not os.path.exists(args.checkpoint_path)):
        print("invalid checkpoint path")
        sys.exit()
    if args.prob is None or args.prob not in prob_names:
        print("invalid target probability")
        sys.exit()
    if args.n_stacks is None or args.n_stacks < 1:
        print("invalid number of stacks")
        sys.exit()
    
    print('input_dim: {}, hidden_dim: {}, layer_num: {}'.format(args.input_dim, args.hidden_dim, args.layer_num))
    print('checkpoint_path: {}'.format(args.checkpoint_path))
    print('activation probablity: {}, # of stacks: {}'.format(args.prob, args.n_stacks))
    print('gpu id: {}'.format(args.gpu))
    IE()
