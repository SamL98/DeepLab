from util import *

if __name__ == '__main__':
    slices = read_slices('slices.pkl')

    for slc in slices:
        for node in slc:
            node.generate_equalized_acc_hist(20)

    save_slices('slices.pkl', slices)