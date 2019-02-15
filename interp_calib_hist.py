import numpy as np
from util import read_slices

slices = read_slices('slices.pkl')

def interp_cluster(cluster):
    hist = cluster.acc_hist
    xs = np.argwhere(hist == 0)
    xp = np.arghwere(hist > 0)
    yp = hist[hist > 0]
	
	if cluster.count_hist[0] == 0 and hist[0] == [0]:
		xp = np.concatenate((0, xp))
		yp = np.concatenate((0, yp))
		
	if cluster.count_hist[-1] == 0 and hist[-1] == 0:
		xp = np.concatenate((xp, len(hist)-1))
		yp = np.concatenate((yp, 1))

    y_interp = np.interp(xs, xp, yp)
    hist[hist == 0] = y_interp
	
	return hist


for slc in slices:
    for cluster in slc:
        cluster.acc_hist[:] = interp_cluster(cluster)
		
save_slices('slices.pkl', slices)