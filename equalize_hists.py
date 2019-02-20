from util import *
import numpy as np

def equalize_hist(count_hist, nb=50):
    bins = list(range(len(count_hist)))

    cdf = count_hist.cumsum()
    cdf = len(count_hist) * cdf / cdf[-1]

    count_hist_interp = np.interp(count_hist, )

if __name__ == '__main__':
    slices = read_slices('slices.pkl')

    for slc in slices:
        for node in slc:
            bin_edges = equalize_hist(node.count_hist)