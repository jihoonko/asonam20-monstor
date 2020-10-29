import argparse
import sys
import os

import torch
import numpy as np
import random
from utils import load_data
from models import MONSTOR
import dgl
import tqdm
import time
import shutil

torch.set_num_threads(1)
graph_names = ['Extended', 'Celebrity', 'WannaCry']
prob_names = ['BT', 'JI', 'LP']
_best_s = {'BT': 3, 'JI': 2, 'LP': 5}

def loss_fn(preds, gt): return torch.mean(torch.abs(gt - preds), dim=1) + (args.lamb * (torch.abs(torch.sum(gt, dim=1) - torch.sum(preds, dim=1)) / torch.sum(gt, dim=1)))

def train():
    global graph_names
    start_time = time.time()
    
    # use only single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "1"
    # re-arange to load graphs
    graph_names = list(filter(lambda x: (x != args.target), graph_names)) + [args.target]
    train_labels = [(graph_names[0], 'train', prob) for prob in prob_names] + [(graph_names[1], 'train', prob) for prob in prob_names]
    val_labels = [(graph_names[0], 'val', prob) for prob in prob_names] + [(graph_names[1], 'val', prob) for prob in prob_names]
    
    print('train_labels: {}'.format(train_labels))
    print('val_labels: {}'.format(val_labels))
    
    g, sX, sy, X, y, _g, _sX, _sy = load_data(train_labels=train_labels, val_labels=val_labels, test_labels=[])
    
    model = MONSTOR(in_feats=args.input_dim, n_hidden=args.hidden_dim, n_layers=args.layer_num).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = './checkpoints/{}_{}_{}_{}_{}'.format(args.target, args.input_dim, args.layer_num, args.hidden_dim, args.lamb)
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    
    # for fixed match size
    batch_size = 20
    bg = {}
    print(g.keys(), _g.keys())
    for lstr, graph in g.items():
        bg[lstr] = dgl.batch([graph for _ in range(batch_size)])
    _bg = {}
    for lstr, graph in _g.items():
        _bg[lstr] = dgl.batch([graph for _ in range(batch_size)])
        
    # settings for preparing minibatch
    trainseq = []
    dataseq = {}
    for tl in train_labels:
        lstr = '_'.join(tl)
        for i in range(0, X[lstr].shape[0], batch_size):
            if i + batch_size > X[lstr].shape[0]:
                break
            trainseq.append((lstr, i))
    for tl in train_labels:
        lstr = '_'.join(tl)
        dataseq[lstr] = np.arange(X[lstr].shape[0])
    
    # for validation
    best_checkpoints = {prob_name: (1e10, 0, 0) for prob_name in prob_names}
    fs = {prob_name: open('{}/val_{}.txt'.format(checkpoint_path, prob_name), 'w') for prob_name in prob_names}
    
    for epoch in range(1, args.epochs+1):
        # adjust learning rate
        for param_group in optimizer.param_groups:
            if epoch <= 10:
                param_group['lr'] = 1e-4 * epoch
            else:
                param_group['lr'] = 1e-2 / epoch
        
        print('Start epoch #{}, elapsed time: {}'.format(epoch, (time.time() - start_time)))
        
        # shuffle minibatches randomly
        for tl in train_labels:
            lstr = '_'.join(tl)
            random.shuffle(dataseq[lstr])
        random.shuffle(trainseq)
        
        model.train()
        for lstr, idx in trainseq:
            optimizer.zero_grad()
            _input = torch.t(X[lstr][dataseq[lstr][idx:(idx+batch_size)]].permute(1, 0, 2).reshape(args.input_dim, -1).cuda())
            gt = y[lstr][dataseq[lstr][idx:(idx+batch_size)]].cuda()
            preds = model(bg[lstr], _input).view(batch_size, -1)
            loss = torch.sum(loss_fn(preds, gt))
            loss.backward()
            optimizer.step()
            
        # save checkpoint for every epoch
        torch.save(model.state_dict(), '{}/{}.pkt'.format(checkpoint_path, epoch))
        
        model.eval()
        with torch.no_grad():
            n_features = args.input_dim
            n_max_stacks = 10
            square_error = {prob_name: np.array([0. for _ in range(n_max_stacks)]) for prob_name in prob_names}
            for vl in val_labels:
                lstr = '_'.join(vl)
                V = _g[lstr].number_of_nodes()
                x = torch.zeros(_bg[lstr].number_of_nodes(), n_features + n_max_stacks, requires_grad=False).cuda()
                for idx in range(0, _sX[lstr].shape[0], batch_size):
                    gt = np.array([torch.sum(_sy[lstr][idx+i]).data for i in range(batch_size)])
                    x[:, n_features-2] = x[:, n_features-1] = torch.t(_sX[lstr][idx:idx+batch_size].view(-1).cuda())
                    for j in range(n_features, n_features + n_max_stacks):
                        x[:, j] = model(_bg[lstr], x[:, j-n_features:j]).data
                        x[:, j-1] = x[:, j] - x[:, j-1]
                        preds = np.array([torch.sum(x[:, j][i*V:(i+1)*V]).data for i in range(batch_size)])
                        square_error[vl[-1]][j - n_features] += np.sum((preds - gt) * (preds - gt))
            for prob_name, _ in fs.items():
                best_s = _best_s[prob_name]
                if (args.target == 'Celebrity') and (prob_name in ['BT', 'LP']): best_s += 2
                best_checkpoints[prob_name] = min(best_checkpoints[prob_name], (square_error[prob_name][best_s - 1], epoch, best_s));
                """
                for i in range(n_max_stacks):
                    best_checkpoints[prob_name] = min(best_checkpoints[prob_name], (square_error[prob_name][i], epoch, i+1))
                """
        model.train()
    
    for prob_name, v in best_checkpoints.items():
        print('for {}: Use {}.pkt (best s = {})'.format(prob_name, v[1], v[2]))
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="choose target graph for masking")
    parser.add_argument("--input-dim", type=int, help="input dimension")
    parser.add_argument("--hidden-dim", type=int, help="hidden dimension")
    parser.add_argument("--gpu", type=int, help="gpu number")
    parser.add_argument("--layer-num", type=int, help="# of layers")
    parser.add_argument("--lamb", type=float, help="hyperparameter for penalty term")
    parser.add_argument("--epochs", type=int, help="number of epochs")
    
    args = parser.parse_args()
    if not args.input_dim: args.input_dim = 4
    
    # argument validation
    if not args.target or args.target not in graph_names:
        print("invalid target graph")
        sys.exit()
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
    if args.lamb is None or args.lamb < 0.:
        print("invalid hyperparameter lambda")
        sys.exit()
    if args.epochs is None or args.epochs < 0:
        print("invalid epoch")
        sys.exit()
        
    print('masked graph: {}'.format(args.target))
    print('input_dim: {}, hidden_dim: {}, layer_num: {}'.format(args.input_dim, args.hidden_dim, args.layer_num))
    print('gpu id: {}'.format(args.gpu))
    print('lambda: {}, epochs: {}'.format(args.lamb, args.epochs))

    train()
