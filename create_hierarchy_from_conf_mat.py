from util import *
from hdf5storage import loadmat
import numpy as np
from util import nc, classes
from create_slices import create_node, create_single_node


def perform_aggregate_clustering(conf_mat):
    slices = []
    node_idx = 0

    while conf_mat.max() > 0:
        r, c = np.unravel_index(np.argmax(conf_mat))
        conf_mat[r,c] = 0

        node = create_node([classes[r], classes[c]], )



if __name__ == '__main__':
    import sys

    assert len(sys.argv) >= 1
    conf_mat_fname = sys.argv[1]

    with open(conf_mat_fname) as f:
        conf_mat = loadmat(conf_mat_fname)['confusion_matrix']