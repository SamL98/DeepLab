'''
Test both the calibration and inference scripts.

Calibration test items:

    0. It ran without errors
    1. # fg pixels is the same across slices
    2. That number is the same as the precomputed num_fg_pix
    3. The # of pixels in each node is the same as the precomputed values
    4. The confidence-adjusted accuracy histograms are all between 0-1

Inference test items:

    0. It ran without errors
    1. The confident predictions have not changed terminal paths.
'''

import numpy as np
from hdf5storage import loadmat, savemat

import os
import shutil
from os.path import join, isfile

import subprocess

'''
Create a fake dataset_info.mat that util will use to return the dataset information
'''
ds_path = 'test_ds'
data_path = join(ds_path, 'node_data')

os.mkdir(ds_path)
os.environ['DS_PATH'] = ds_path

classes = ['background', 'object', 'being', 'void']
n_val, n_test = 5, 5
img_size = 5
nc = len(classes)-1

dataset_info = {
    'class_labels': classes,
    'n_val': n_val,
    'n_test': n_test,
    'img_size': img_size,
    'nc': nc
}

savemat(join(ds_path, 'dataset_info.mat'), dataset_info)

'''
Pixel counts used later for validating calibration
'''
num_fg_pix = 0
per_class_counts = {}

from util import *

'''
Create a dummy ground truth or logit dataset
'''
def create_dummy_dataset(ds_type):
    global ds_path, dataset_info
    global num_fg_pix, per_class_counts

    nc = dataset_info['nc']
    img_size = dataset_info['img_size']

    ds_type = ds_type.lower()
    os.mkdir(join(ds_path, ds_type))

    for imset in ['val', 'test']:
        os.mkdir(join(ds_path, ds_type, imset))

        n_ex = eval('n_%s' % imset)
        for ex_idx in range(1, n_ex+1):
            if ds_type == 'truth':
                fmt = imset + '_%06d_pixeltruth.mat' % ex_idx
                name = 'truth_img'
                ex = np.random.choice(range(nc), (dataset_info['img_size'], dataset_info['img_size']), replace=True)
                ex = ex.astype(np.uint8)
            else:
                fmt = imset + '_%06d_logits.mat' % ex_idx
                name = 'logits_img'
                ex = np.random.randn(img_size, img_size, nc)

                pred = np.argmax(ex[:,1:], -1).ravel() + 1
                num_fg_pix += (pred > 0).sum()

                for lab in np.unique(pred):
                    class_sum = (pred == lab).sum()
                    if not lab in per_class_counts:
                        per_class_counts[lab] = class_sum
                    else:
                        per_class_counts[lab] += class_sum

            savemat(join(ds_path, ds_type, imset, fmt), { name: ex })

def setup():
    print('Setting up test environment.')

    global data_path
    os.mkdir(data_path)

    for imset in ['val', 'test']:
        fname = f'{imset}_ordered.txt'
        if isfile(fname):
            os.rename(fname, fname+'.bkp')

    subdirs = ['truth', 'deeplab_prediction']
    for subdir in subdirs:
        create_dummy_dataset(subdir)

def create_slices():
    print('Creating test slice file.')

    from create_slices import create_node, create_single_node

    slices = [[create_single_node(class_lab) for class_lab in classes]]
    slices.append([create_node(['being', 'object'], 'foreground')])

    global ds_path
    save_slices(join(ds_path, 'test_slices.pkl'), slices)

def calibrate(sm_by_slice):
    print('Performing calibration on dummy dataset.')

    global ds_path, data_path

    args_dict = {
        'slice_file': join(ds_path, 'test_slices.pkl'),
        'imset': 'val',
        'num_proc': 2,
        'sigma': 0.1,
        'alpha': 0.05,
        'nb': 10,
        'output_file': join(ds_path, 'test_slices_output.pkl'),
        'data_dir': data_path
    }
    args = list(map(lambda kv: f'--{kv[0]}={kv[1]}', args_dict.items()))
    
    if sm_by_slice:
        args.append('--sm_by_slice')

    retcode = subprocess.call(['python', 'do_calib.py'] + args)
    assert retcode == 0, 'Non-zero return code from calibration: %d' % retcode

def validate_calibration():
    print('Validating dummy calibration.')

    global ds_path

    ''' Validate item #1 '''
    slices = read_slices(join(ds_path, 'test_slices_output.pkl'))

    slc0_num_fg_pix = sum([node.tot_hist.sum() for node in slices[0]])
    for i, slc in enumerate(slices[1:]):
        slc_num_fg_pix = sum([node.tot_hist.sum() for node in slc])
        perc_diff = abs(slc_num_fg_pix-slc0_num_fg_pix) / float(slc0_num_fg_pix)
        assert perc_diff < 1e-3, 'Non-consistent num_fg_pix across slices: %d vs %d for slice %d' % (slc0_num_fg_pix, slc_num_fg_pix, i)

    ''' Validate item #2 '''
    global num_fg_pix
    perc_diff = abs(num_fg_pix - slc0_num_fg_pix) / num_fg_pix
    assert perc_diff < 1e-3, 'Non-consistent num_fg_pix on slice 0: %d vs %d' % (num_fg_pix, slc0_num_fg_pix)

    ''' Validate item #3 '''
    global per_class_counts
    for slc in slices:
        for node in slc:
            init_num_fg_pix = per_class_counts[node.node_idx]
            final_num_fg_pix = node.tot_hist.sum()
            perc_diff = abs(init_num_fg_pix-final_num_fg_pix)/init_num_fg_pix
            assert perc_diff < 1e-3, 'Non-consistent num_fg_pix for %s node: %d vs %d' % (node.name, init_num_fg_pix, final_num_fg_pix)

    ''' Validate item #4 '''
    for slc in slices:
        for node in slc:
            conf_adj_acc_hist = node.get_conf_acc_hist()
            num_out_of_range = ((conf_adj_acc_hist < 0) | (conf_adj_acc_hist > 1)).sum()
            assert num_out_of_range == 0, 'Confidence values out of range for %s node: %s' % (node.name, conf_adj_acc_hist.__repr__())

def infer():
    print('Performing dummy inference.')

    global ds_path

    name = 'dummy_calib'
    os.mkdir(join(ds_path, 'deeplab_prediction', 'test', name))

    args_dict = {
        'slice_file': join(ds_path, 'test_slices_output.pkl'),
        'imset': 'test',
        'num_proc': 2,
        'conf_thresh': 0.75
    }
    args = list(map(lambda kv: f'--{kv[0]}={kv[1]}', args_dict.items()))
    
    if sm_by_slice:
        args.append('--sm_by_slice')

    retcode = subprocess.call(['python', 'do_inf.py'] + args)
    assert retcode == 0, 'Non-zero return code from inference: %d' % retcode

def validate_inference():
    print('Validating dummy inference.')

    global ds_path, dataset_info

    ''' Validate item #1 '''
    slices = read_slices(join(ds_path, 'test_slices_output.pkl'))
    name = 'dummy_calib'

    for idx in range(1, dataset_info['n_test']+1):
        logits = load_logits('test', idx, reshape=False)
        calib_pred = load_calib_pred('test', idx, 0.75, name).ravel()
        dl_pred = (np.argmax(logits[:,1:], -1) + 1).ravel()

        for calib_lab, dl_lab in zip(calib_pred, dl_pred):
            assert is_in_gt_path(calib_lab, dl_lab, slices), 'Calibrated prediction jumped paths: %d -> %d' % (dl_lab, calib_lab)

def teardown():
    global ds_path
    os.environ['DS_PATH'] = None
    shutil.rmtree(ds_path)

    for imset in ['val', 'test']:
        fname = f'{imset}_ordered.txt'
        if isfile(fname+'.bkp'):
            os.rename(fname+'.bkp', fname)

if __name__ == '__main__':
    sm_by_slice = False

    setup()
    create_slices()
    calibrate(sm_by_slice)
    validate_calibration()
    infer(sm_by_slice)
    validate_inference()
    teardown()