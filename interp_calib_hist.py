import numpy as np
from util import read_slices

slices = read_slices('slices.pkl')

def interp_cluster(cluster):
    hist = cluster.acc_hist
    xs = np.argwhere(hist == 0)
    xp = np.arghwere(hist > 0)
    yp = hist[hist > 0]

    y_interp = np.interp(xs, xp, yp, left=)
    hist[hist == 0] = y_interp


for slc in slices:
    for cluster in slc:
        interp_cluster(cluster)