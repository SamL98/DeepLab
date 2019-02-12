from hdf5storage import loadmat, savemat

ds_info = loadmat('D:/datasets/processed/voc2012/dataset_info.mat')
classes = ds_info['class_labels'][1:-1]

def create_cluster(labels):
	global classes
	clust = []
	for label in labels:
		clust.append(classes.index(label))
	return clust

slices = [[[i] for i in range(len(classes))]]

slc = []
slc.append(create_cluster(['cow', 'sheep', 'horse']))
slc.append([classes.index('dog')])
slc.append([classes.index('cat')])
slc.append([classes.index('bird')])
slc.append([classes.index('person')])
slc.append(create_cluster(['car', 'bus', 'motorbike', 'bicycle', 'train']))
slc.append([classes.index('boat')])
slc.append([classes.index('aeroplane')])
slc.append([classes.index('bottle')])
slc.append([classes.index('tvmonitor')])
slc.append([classes.index('potted plant')])
slc.append(create_cluster(['chair', 'sofa']))
slc.append([classes.index('diningtable')])
slices.append(slc[:])

slc = []
slc.append(create_cluster(['cow', 'sheep', 'horse', 'dog', 'cat']))
slc.append([classes.index('bird')])
slc.append([classes.index('person')])
slc.append(create_cluster(['car', 'bus', 'motorbike', 'bicycle', 'train', 'boat', 'aeroplane']))
slc.append([classes.index('bottle')])
slc.append([classes.index('tvmonitor')])
slc.append([classes.index('potted plant')])
slc.append(create_cluster(['chair', 'sofa', 'diningtable']))
slices.append(slc[:])

slc = []
slc.append(create_cluster(['cow', 'sheep', 'horse', 'dog', 'cat', 'bird', 'person']))
slc.append(create_cluster(['car', 'bus', 'motorbike', 'bicycle', 'train', 'boat', 'aeroplane', 'bottle', 'tvmonitor', 'potted plant', 'chair', 'sofa', 'diningtable']))
slices.append(slc[:])

savemat('slices.mat', {'slices': slices})