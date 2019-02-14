from hdf5storage import loadmat
import pickle
from collections import namedtuple
import numpy as np

Cluster = namedtuple('Cluster', 'name cluster_idx terminals corr_hist count_hist acc_hist')

ds_info = loadmat('D:/datasets/processed/voc2012/dataset_info.mat')
classes = ds_info['class_labels'][:-1]

cluster_idx = 1
res = .05
nb = int(1./res)

def create_cluster(labels, name):
	global classes, cluster_idx, nb

	terminals = []
	for label in labels:
		terminals.append(classes.index(label))

	clust =  Cluster(name, 
					cluster_idx, 
					terminals, 
					np.zeros((nb), dtype=np.uint64), 
					np.zeros((nb), dtype=np.uint64), 
					np.zeros((nb), dtype=np.float32))
	cluster_idx += 1
	return clust

def create_single_cluster(label):
	return create_cluster([label], label)

slices = [[create_single_cluster(classes[i]) for i in range(1, len(classes))]]

slc = []
slc.append(create_cluster(['cow', 'sheep', 'horse'], 'ungulate'))
slc.append(create_single_cluster('dog'))
slc.append(create_single_cluster('cat'))
slc.append(create_single_cluster('bird'))
slc.append(create_single_cluster('person'))
slc.append(create_cluster(['car', 'bus', 'motorbike', 'bicycle', 'train'], 'ground transportation'))
slc.append(create_single_cluster('boat'))
slc.append(create_single_cluster('aeroplane'))
slc.append(create_single_cluster('bottle'))
slc.append(create_single_cluster('tvmonitor'))
slc.append(create_single_cluster('potted plant'))
slc.append(create_cluster(['chair', 'sofa'], 'seating'))
slc.append(create_single_cluster('diningtable'))
slices.append(slc[:])

slc = []
slc.append(create_cluster(['cow', 'sheep', 'horse', 'dog', 'cat'], 'land animal'))
slc.append(create_single_cluster('bird'))
slc.append(create_single_cluster('person'))
slc.append(create_cluster(['car', 'bus', 'motorbike', 'bicycle', 'train', 'boat', 'aeroplane'], 'vehicle'))
slc.append(create_single_cluster('bottle'))
slc.append(create_single_cluster('tvmonitor'))
slc.append(create_single_cluster('potted plant'))
slc.append(create_cluster(['chair', 'sofa', 'diningtable'], 'furniture'))
slices.append(slc[:])

slc = []
slc.append(create_cluster(['cow', 'sheep', 'horse', 'dog', 'cat', 'bird', 'person'], 'being'))
slc.append(create_cluster(['car', 'bus', 'motorbike', 'bicycle', 'train', 'boat', 'aeroplane', 'bottle', 'tvmonitor', 'potted plant', 'chair', 'sofa', 'diningtable'], 'object'))
slices.append(slc[:])

with open('slices.pkl', 'wb') as f:
	pickle.dump(slices, f)