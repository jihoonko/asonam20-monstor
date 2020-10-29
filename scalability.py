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

def evaluation():
    # use only single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Load graphs from .txt file
    with open(args.graph_path, "r") as f:
        V, E = map(int, f.readline().split())
        tokens = [line.strip().split() for line in f]
        srcs, dsts, probs = zip(*tokens)
        srcs, dsts, probs = list(map(int, srcs)), list(map(int, dsts)), list(map(float, probs))
    graph = dgl.DGLGraph()
    graph.add_nodes(V)
    graph.add_edges(srcs, dsts)
    graph.edata['weight'] = torch.FloatTensor(probs).cuda()
    
    # Load pretrained weights
    model = MONSTOR(in_feats=args.input_dim, n_hidden=args.hidden_dim, n_layers=args.layer_num).cuda()
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    
    batch_size = 1
    bg = dgl.batch([graph for _ in range(batch_size)])
    
    # measure the runtime of 1,000 influence evaluations
    with torch.no_grad():
        total_elapsed_time = 0.
        for epoch in range(20):
            # generate 50 random inputs
            test_tensor = torch.zeros(50, bg.number_of_nodes(), args.input_dim).cuda()
            for i in range(50):
                test_tensor[i, torch.randperm(bg.number_of_nodes())[:np.random.randint(1, bg.number_of_nodes() // 10)], (args.input_dim - 2):] = 1.
            # measure the runtime of 50 influence evaluations using time.perf_counter()
            start_time = time.perf_counter()
            for i in range(50):
                _ = model(bg, test_tensor[i])
            end_time = time.perf_counter()
            total_elapsed_time += (end_time - start_time)
            del test_tensor
        print('elapsed time: {} s.'.format(total_elapsed_time))    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", help="path of graph for evaluation")
    parser.add_argument("--input-dim", type=int, help="input dimension")
    parser.add_argument("--hidden-dim", type=int, help="hidden dimension")
    parser.add_argument("--layer-num", type=int, help="# of layers")
    parser.add_argument("--checkpoint-path", help="path of target checkpoint")
    parser.add_argument("--gpu", type=int, help="gpu number")
    
    args = parser.parse_args()
    if not args.input_dim: args.input_dim = 4
    
    # argument validation
    if args.graph_path is None or (not os.path.exists(args.graph_path)):
        print("invalid graph path")
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
    if args.checkpoint_path is None or (not os.path.exists(args.checkpoint_path)):
        print("invalid checkpoint path")
        sys.exit()
    
    print('graph')
    print('input_dim: {}, hidden_dim: {}, layer_num: {}'.format(args.input_dim, args.hidden_dim, args.layer_num))
    print('checkpoint_path: {}'.format(args.checkpoint_path))
    print('gpu id: {}'.format(args.gpu))
    evaluation()
