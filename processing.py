import sys
import numpy as np
import gzip
import scipy.sparse as sp
import pickle
import random
from tqdm import tqdm

class convert:
    def __init__(self):
        pass

    def makedata(self, graph_name, start_idx, n, feat=4, eps=0.001):
        self.X, self.y = [], []
        self.sX, self.sy = [], []
        print("[INFO] processing {}".format(graph_name))
        for idx in tqdm(range(start_idx, start_idx + n)):
            try:
                with open('raw_data/{}/{}.txt'.format(graph_name, idx), 'rb') as f:
                    data = np.array([list(map(float, line.strip().split())) for line in f])
                    data = np.vstack((np.zeros((feat-1, data.shape[1])), data))
                    for i in range(50):
                        # generate data for each step
                        X, y = np.copy(data[i:i+feat,:]), np.copy(data[i+feat,:])
                        for j in range(feat-1):
                            X[j, :] = X[j+1, :] - X[j, :]
                        if np.sum(y) - np.sum(X[feat-1, :]) < eps: break # if simulation is almost ended
                        self.X.append(X)
                        self.y.append(y)
                    self.sX.append(data[feat-1].copy())
                    self.sy.append(data[-1].copy())
                    del data
            except:
                break
        self.X, self.y = np.array(self.X, dtype=np.float32), np.array(self.y, dtype=np.float32)
        self.sX, self.sy = np.array(self.sX, dtype=np.float32), np.array(self.sy, dtype=np.float32)

if __name__ == '__main__':
    data = convert()

    dataset_name = sys.argv[1].strip().split('_')
    if len(dataset_name) != 3:
        print("Assertion failed: invalid dataset_name format")
        sys.exit(1)
    if dataset_name[0] not in ['Extended', 'Celebrity', 'WannaCry']:
        print("Assertion failed: invalid graph name")
        sys.exit(1)
    if dataset_name[1] not in ['train', 'test']:
        print("Assertion failed: input should be train or test")
        sys.exit(1)
    if dataset_name[2] not in ['BT', 'JI', 'LP']:
        print("Assertion failed: invalid activation probability")
        sys.exit(1)
    
    if dataset_name[1] == 'train':
        data.makedata(sys.argv[1].strip(), 1, 1600)
        for var in ["X", "y", "sX", "sy"]:
            with gzip.open('datadir/{}_train_{}_{}_random.pkl.gz'.format(dataset_name[0], dataset_name[2], var), 'wb') as f_out:
                pickle.dump(getattr(data, var), f_out, protocol=4)

        data.makedata(sys.argv[1].strip(), 1601, 400)
        for var in ["sX", "sy"]:
            with gzip.open('datadir/{}_val_{}_{}_random.pkl.gz'.format(dataset_name[0], dataset_name[2], var), 'wb') as f_out:
                pickle.dump(getattr(data, var), f_out, protocol=4)
        
        data.makedata(sys.argv[1].strip(), 2001, 1600)
        for var in ["X", "y", "sX", "sy"]:
            with gzip.open('datadir/{}_train_{}_{}_degree.pkl.gz'.format(dataset_name[0], dataset_name[2], var), 'wb') as f_out:
                pickle.dump(getattr(data, var), f_out, protocol=4)

        data.makedata(sys.argv[1].strip(), 3601, 400)
        for var in ["sX", "sy"]:
            with gzip.open('datadir/{}_val_{}_{}_degree.pkl.gz'.format(dataset_name[0], dataset_name[2], var), 'wb') as f_out:
                pickle.dump(getattr(data, var), f_out, protocol=4)
    elif dataset_name[1] == 'test':
        data.makedata(sys.argv[1].strip(), 1, 2000)
        for var in ["X", "y", "sX", "sy"]:
            with gzip.open('datadir/{}_test_{}_{}_random.pkl.gz'.format(dataset_name[0], dataset_name[2], var), 'wb') as f_out:
                pickle.dump(getattr(data, var), f_out, protocol=4)

        data.makedata(sys.argv[1].strip(), 2001, 2000)
        for var in ["X", "y", "sX", "sy"]:
            with gzip.open('datadir/{}_test_{}_{}_degree.pkl.gz'.format(dataset_name[0], dataset_name[2], var), 'wb') as f_out:
                pickle.dump(getattr(data, var), f_out, protocol=4)

