import numpy as np
import pickle as pkl
import gzip

import torch
from dgl import DGLGraph
import networkx as nx

def _load_data_inner(train, val, test):
    targets = {'train': train, 'val': val, 'test': test}
    objs = {'train': [], 'val': [], 'test': []}
    
    for mark in ('train', 'val', 'test'):
        for name in targets[mark]:
            print('Load data for {}: {}'.format(mark, name))
            objects = []
            with gzip.open('./datadir/{}_{}_{}_graph.pkl.gz'.format(name[0], name[1], name[2]), 'rb') as f:
                objects.append(pkl.load(f))
            labels = ['X', 'y', 'sX', 'sy'] if (mark == 'train') else ['sX', 'sy']
            for label in labels:
                with gzip.open('./datadir/{}_{}_{}_{}_random.pkl.gz'.format(name[0], name[1], name[2], label), 'rb') as f:
                    seeds_random = pkl.load(f)
                with gzip.open('./datadir/{}_{}_{}_{}_degree.pkl.gz'.format(name[0], name[1], name[2], label), 'rb') as f:
                    seeds_degree = pkl.load(f)
                print(seeds_random.shape, seeds_degree.shape)
                objects.append(np.concatenate((seeds_random, seeds_degree), axis=0))
            objs[mark].append(tuple(objects))
    return objs['train'], objs['val'], objs['test']

def load_data(train_labels, val_labels, test_labels):
    g, sX, sy, X, y = {}, {}, {}, {}, {}
    _g, _sX, _sy = {}, {}, {}
    train_data, val_data, test_data = _load_data_inner(train=train_labels, val=val_labels, test=test_labels)
    for label, data in zip(train_labels, train_data):
        lstr = '_'.join(label)
        print('train: load {}...'.format(lstr))

        g[lstr] = DGLGraph()
        g[lstr].from_scipy_sparse_matrix(data[0])
        g[lstr].edata['weight'] = torch.from_numpy(np.float32(data[0].data)).cuda()
        
        X[lstr]  = torch.FloatTensor(data[1])  
        y[lstr]  = torch.FloatTensor(data[2])
        sX[lstr] = torch.FloatTensor(data[3])
        sy[lstr] = torch.FloatTensor(data[4])

    for label, data in zip(val_labels, val_data):
        lstr = '_'.join(label)
        print('val: load {}...'.format(lstr))

        _g[lstr] = DGLGraph()
        _g[lstr].from_scipy_sparse_matrix(data[0])
        _g[lstr].edata['weight'] = torch.from_numpy(np.float32(data[0].data)).cuda()

        _sX[lstr] = torch.FloatTensor(data[1])
        _sy[lstr] = torch.FloatTensor(data[2])
        
    for label, data in zip(test_labels, test_data):
        lstr = '_'.join(label)
        print('test: load {}...'.format(lstr))

        _g[lstr] = DGLGraph()
        _g[lstr].from_scipy_sparse_matrix(data[0])
        _g[lstr].edata['weight'] = torch.from_numpy(np.float32(data[0].data)).cuda()

        _sX[lstr] = torch.FloatTensor(data[1])
        _sy[lstr] = torch.FloatTensor(data[2])
        
    
    return g, sX, sy, X, y, _g, _sX, _sy

    
