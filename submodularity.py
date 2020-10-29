import argparse
import sys
import os

import torch
import numpy as np
from utils import load_data
from models import MONSTOR
import dgl
import time
import tqdm
import pickle

torch.set_num_threads(4)
graph_names = ['Extended', 'Celebrity', 'WannaCry']
prob_names = ['BT', 'JI', 'LP']

def run(model, target_graph, init_state, n_stacks):
    n_features = args.input_dim
    _V = target_graph.number_of_nodes()
    idxseq = [0, 1, 2, 3, 0, 1, 2, 3]
    now, prv = -1, -1   
    with torch.no_grad():
        x = torch.zeros(_V, n_features, requires_grad=False).cuda()
        x[:, n_features-1] = x[:, 0] = init_state
        for j in range(1, n_stacks+1):
            now, prv = j%4, (j+3)%4
            x[:, now] = model(target_graph, x[:, idxseq[now:now+4]]).data
            x[:, prv] = x[:, now] - x[:, prv]
            if torch.sum(x[:, prv]) < 1e-6:
                break
        return torch.sum(x[:, now]).item()

def evaluation():
    global graph_names
    
    # use only single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    test_labels = [(graph_name, 'test', args.prob) for graph_name in graph_names]
    print('test_labels: {}'.format(test_labels))
    
    g, sX, sy, X, y, _g, _sX, _sy = load_data(train_labels=[], val_labels=[], test_labels=test_labels)
    model = MONSTOR(in_feats=args.input_dim, n_hidden=args.hidden_dim, n_layers=args.layer_num).cuda()
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    
    sample_n = 5000
    
    with torch.no_grad():
        for tl in test_labels:
            graph = _g['_'.join(tl)]
            V, E = graph.number_of_nodes(), graph.number_of_edges()
            
            pairs, seeds = [], []
            for _ in tqdm.trange(sample_n):
                # generate random seed
                _A = torch.randperm(V)[:np.random.randint(V // 10)]
                _B = torch.randperm(V)[:np.random.randint(V // 10)]
                A, B = torch.zeros(V).bool().cuda(), torch.zeros(V).bool().cuda()
                A[_A] = True
                B[_B] = True
                assert(torch.max(A.float()).item() < 1.00001)
                assert(torch.max(B.float()).item() < 1.00001)
                assert(torch.max((A+B).float()).item() < 1.00001)
                assert(torch.max((A*B).float()).item() < 1.00001)
                assert(A.float().sum() <= V / 10)
                assert(B.float().sum() <= V / 10)
                assert(np.allclose(A.float().sum().item() + B.float().sum().item(),
                                  (A + B).float().sum().item() + (A * B).float().sum().item()))
                
                pairs.append((run(model, graph, (A * B).float(), args.n_stacks) + run(model, graph, (A + B).float(), args.n_stacks),
                              run(model, graph, A.float(), args.n_stacks) + run(model, graph, B.float(), args.n_stacks)))
                seeds.append((_A.data.cpu().numpy(), _B.data.cpu().numpy()))
            
            # compute accuracy and MAPE
            cnt, mape_sum = 0, 0.
            for lhs, rhs in pairs:
                if lhs > rhs:
                    cnt += 1
                    mape_sum += abs(lhs - rhs) / rhs
            print('for target graph {} | Accuracy: {}, MAPE: {}'.format(tl[0], (len(pairs) - cnt) / len(pairs), mape_sum / (cnt + 1e-9)))
            
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
    evaluation()
