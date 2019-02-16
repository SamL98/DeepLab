from hdf5storage import loadmat
import numpy as np
from os.path import join

from util import num_img_for, load_gt, fg_mask_for

def order_imset_by_num_fg(imset, save=False):
    num_img = num_img_for(imset)
    num_fg_list = []

    for idx in range(1, num_img+1):
        gt = load_gt(imset, idx)
        fg_count = fg_mask_for(gt).sum()
        num_fg.append(fg_count)

    ordered_idxs = list(reversed(np.argsort(np.array(num_fg))+1))

    if save:
        fname = imset.lower() + '_ordered.txt'
        with open(fname, 'w') as f:
            f.write('\n'.join([str(idx) for idx in ordered_num_fg]))

    return ordered_idxs

if __name__ == '__main__':
    import sys

    imset = 'val'
    if len(sys.argv) > 1:
        imset = sys.argv[1]

    _ = order_imset_by_num_fg(imset, save=True)