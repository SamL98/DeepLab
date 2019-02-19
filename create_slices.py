from hdf5storage import loadmat
from util import *

node_idx = 1
res = .001
nb = int(1./res)

def create_node(labels, name):
	global node_idx, nb

	terminals = []
	for label in labels:
		terminals.append(classes.index(label))

	clust = Node(name, node_idx, terminals, nb)
	node_idx += 1
	return clust

def create_single_node(label):
	return create_node([label], label)


if __name__ == '__main__':
	slices = [[create_single_node(classes[i]) for i in range(1, len(classes))]]

	slc = []
	slc.append(create_node(['cow', 'sheep', 'horse'], 'ungulate'))
	slc.append(create_single_node('dog'))
	slc.append(create_single_node('cat'))
	slc.append(create_single_node('bird'))
	slc.append(create_single_node('person'))
	slc.append(create_node(['car', 'bus', 'motorbike', 'bicycle', 'train'], 'ground transportation'))
	slc.append(create_single_node('boat'))
	slc.append(create_single_node('aeroplane'))
	slc.append(create_single_node('bottle'))
	slc.append(create_single_node('tvmonitor'))
	slc.append(create_single_node('potted plant'))
	slc.append(create_node(['chair', 'sofa'], 'seating'))
	slc.append(create_single_node('diningtable'))
	slices.append(slc[:])

	slc = []
	slc.append(create_node(['cow', 'sheep', 'horse', 'dog', 'cat'], 'land animal'))
	slc.append(create_single_node('bird'))
	slc.append(create_single_node('person'))
	slc.append(create_node(['car', 'bus', 'motorbike', 'bicycle', 'train', 'boat', 'aeroplane'], 'vehicle'))
	slc.append(create_single_node('bottle'))
	slc.append(create_single_node('tvmonitor'))
	slc.append(create_single_node('potted plant'))
	slc.append(create_node(['chair', 'sofa', 'diningtable'], 'furniture'))
	slices.append(slc[:])

	slc = []
	slc.append(create_node(['cow', 'sheep', 'horse', 'dog', 'cat', 'bird', 'person'], 'being'))
	slc.append(create_node(['car', 'bus', 'motorbike', 'bicycle', 'train', 'boat', 'aeroplane', 'bottle', 'tvmonitor', 'potted plant', 'chair', 'sofa', 'diningtable'], 'object'))
	slices.append(slc[:])

	save_slices('slices.pkl', slices)