import numpy as np
from hdf5storage import loadmat, savemat

import os
from os.path import join

os.environ['DS_PATH'] = '.'
savemat('dataset_info.mat', {
    'class_labels': ['background', 'foreground']
})

from util import *

num_img = 2
img_size = 5
nc = 2

def setup():
    global num_img, img_size, nc

    os.mkdir('calib_test')

    for idx in range(1, num_img+1):
        logits = np.random.rand((img_size, img_size, nc), dtype=np.float32)
        logits[:,::2,0] = 1
        logits[::2,:,1] = 0
        savemat('calib_test/calib_test_%06d_logits.mat' % idx, {
            'logits_img': logits
        })

def create_slices():
    from create_slices import create_node   

def calibrate():


def teardown():
    os.environ['DS_PATH'] = None
    os.remove('calib_test')
    os.remove('dataset_info.mat')

if __name__ == '__main__':
    setup()
    create_slices()
    calibrate()
    teardown()