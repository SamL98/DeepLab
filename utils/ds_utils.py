from os.path import join, isfile
import os

ds_path = 'D:/datasets/processed/voc2012'

if 'DS_PATH' in os.environ:
	ds_path = os.environ['DS_PATH']

ds_info_fname = join(ds_path, 'dataset_info.mat')

if isfile(ds_info_fname):
	from hdf5storage import loadmat
	ds_info = loadmat(ds_info_fname)

	classes = ds_info['class_labels'][1:-1]
	#nc = len(classes)-1
	nc = len(classes)

	num_val = None
	if 'n_val' in ds_info:
		num_val = ds_info['n_val']

	num_test = None
	if 'n_test' in ds_info:
		num_test = ds_info['n_test']

def num_img_for(imset):
	global num_val, num_test
	if num_val and num_test:
		return eval('num_%s' % imset.lower())

	val_size = 724
	if imset == 'val':
		return val_size
	else:
		return 1449-val_size

def imset_iter(imset):
	return range(1, num_img_for(imset)+1)
