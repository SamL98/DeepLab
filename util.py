import pickle
from collections import namedtuple
Cluster = namedtuple('Cluster', 'name cluster_idx terminals corr_hist count_hist acc_hist')

def read_slices(fname):
    with open(fname, 'rb') as f:
        slices = pickle.load(f)
    return slices