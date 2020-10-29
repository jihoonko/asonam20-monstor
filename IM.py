import argparse
import sys
import os

import torch
import numpy as np
from utils import load_data
from models import MONSTOR
import dgl
import dgl.function as fn
from queue import PriorityQueue
from dgl import DGLGraph
import time

torch.set_num_threads(4)
graph_names = ['Extended', 'Celebrity', 'WannaCry']
prob_names = ['BT', 'JI', 'LP']
model = None

def eval_batch(target_graph, init_state, batch_size, n_stacks):
    n_features = 4
    _V = target_graph.number_of_nodes() // batch_size
    idxseq = [0, 1, 2, 3, 0, 1, 2, 3]
    now, prv = -1, -1   
    with torch.no_grad():
        x = torch.zeros(target_graph.number_of_nodes(), n_features, requires_grad=False).cuda()
        x[:, n_features-1] = x[:, 0] = init_state
        for j in range(1, n_stacks+1):
            now, prv = j%4, (j+3)%4
            x[:, now] = model(target_graph, x[:, idxseq[now:now+4]]).data
            x[:, prv] = x[:, now] - x[:, prv]
            if torch.sum(x[:, prv]) < 1e-6 * batch_size:
                break
        return [torch.sum(x[i:(i+_V), now]).item() for i in range(0, _V * batch_size, _V)]

def celf(target_graph, batch_size, n_stacks):
    s = time.time()
    V = target_graph.number_of_nodes()
    
    delta = [1e10 for _ in range(V)]
    update_time = [-1 for _ in range(V)]
    que = PriorityQueue(maxsize=V+1)
    for i in range(V):
        que.put((-delta[i], i))
    
    batched_graph = dgl.batch([target_graph for _ in range(batch_size)])
    cnt, prv_mc, ans, curr = 0, 0, [], torch.zeros(V * batch_size)
    
    while (not que.empty()) and cnt < 100:
        _, idx = que.get()
        if update_time[idx] == cnt:
            prv_mc += delta[idx]
            ans.append(idx)
            for i in range(batch_size):
                curr[i * V + idx] = 1.
            # print((time.time() - s), idx)
            cnt += 1
        else:
            idxs = [idx] + [que.get()[1] for _ in range(1, batch_size)]
            tmp_state = curr.clone().detach()
            for i in range(batch_size):
                tmp_state[i * V + idxs[i]] = 1.
            nxt_delta = eval_batch(batched_graph, tmp_state, batch_size, n_stacks)
            # print('%s: %s -> %s' % (idx, delta[idx], nxt_delta))
            for i in range(batch_size):
                update_time[idxs[i]], delta[idxs[i]] = cnt, (nxt_delta[i] - prv_mc)
                que.put((-delta[idxs[i]], idxs[i]))
            
    # monitor(_print=True)
    return ans

def ublf(target_graph, eps, batch_size, n_stacks):
    s = time.time()
    graph = DGLGraph()
    V, E = target_graph.number_of_nodes(), target_graph.number_of_edges() 
    graph.add_nodes(V)
    
    # Flip the direction of edges
    graph.add_edges(target_graph.edges()[1], target_graph.edges()[0], {'weight': target_graph.edata['weight']})
    graph.ndata['ub'] = torch.ones(V).cuda()
    graph.ndata['a'] = torch.ones(V).cuda()
    
    # Compute UBound
    while True:
        graph.update_all(message_func=fn.u_mul_e('a', 'weight', 'msg'),
                         reduce_func=fn.sum(msg='msg', out='ub_delta'))
        ub_delta = graph.ndata.pop('ub_delta')
        graph.ndata['ub'] += ub_delta
        graph.ndata['a'] = ub_delta
        if torch.sum(ub_delta).item() < eps:
            break
    
    delta = graph.ndata.pop('ub').data.cpu().numpy()
    
    update_time = [-1 for _ in range(V)]
    que = PriorityQueue(maxsize=V+1)
    for i in range(V):
        que.put((-delta[i], i))
    
    batched_graph = dgl.batch([target_graph for _ in range(batch_size)])
    cnt, prv_mc, ans, curr = 0, 0, [], torch.zeros(V * batch_size)
    
    while (not que.empty()) and cnt < 100:
        _, idx = que.get()
        if update_time[idx] == cnt:
            prv_mc += delta[idx]
            ans.append(idx)
            for i in range(batch_size):
                curr[i * V + idx] = 1.
            cnt += 1
        else:
            idxs = [idx] + [que.get()[1] for _ in range(1, batch_size)]
            tmp_state = curr.clone().detach()
            for i in range(batch_size):
                tmp_state[i * V + idxs[i]] = 1.
            nxt_delta = eval_batch(batched_graph, tmp_state, batch_size, n_stacks)
            for i in range(batch_size):
                update_time[idxs[i]], delta[idxs[i]] = cnt, (nxt_delta[i] - prv_mc)
                que.put((-delta[idxs[i]], idxs[i]))
    return ans

def IM():
    global graph_names
    global model
    
    # use only single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    test_labels = [(graph_name, 'test', args.prob) for graph_name in graph_names]
    print('test_labels: {}'.format(test_labels))
    
    g, sX, sy, X, y, _g, _sX, _sy = load_data(train_labels=[], val_labels=[], test_labels=test_labels)
    model = MONSTOR(in_feats=args.input_dim, n_hidden=args.hidden_dim, n_layers=args.layer_num).cuda()
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    
    with torch.no_grad():
        for tl in test_labels:
            lstr = '_'.join(tl)
            selected = None
            if args.prob == 'LP': # since UBLF cannot used for LP
                selected = celf(_g[lstr], 20, args.n_stacks)
            else:
                selected = ublf(_g[lstr], 1e-6, 20, args.n_stacks)
            print("for target graph {} | Selected nodes are {}".format(tl[0], ' '.join(map(str, selected))))
    
        
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
    if type(args.input_dim) != int or args.input_dim < 2 or args.input_dim > 4:
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
    IM()
