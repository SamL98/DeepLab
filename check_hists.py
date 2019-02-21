from util import *

if __name__ == '__main__':
    imset = 'val'
    num_img = num_img_for(imset)
    num_fg_pix = 0

    for idx in range(1, num_img+1):
        gt = load_gt(imset, idx, reshape=True)
        num_fg_pix += fg_mask_for(gt).sum()

    slices = read_slices('slices.pkl')

    for slc in slices:
        slc_fg_pix = 0

        for node in slc:
            slc_fg_pix += node.get_fg_count()

        assert slc_fg_pix == num_fg_pix