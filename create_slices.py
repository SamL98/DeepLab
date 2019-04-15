from hdf5storage import loadmat
import util
from node import Node
from slc import Slice

node_idx = 1
res = .005
nb = int(1./res)

def create_node(labels, name, prev_slc_labels):
	global node_idx, nb

	children = []
	for label in labels:
		children.append(prev_slc_labels.index(label))

	node = Node(name, node_idx, sorted(children), 100)
	node_idx += 1
	return node

def create_single_node(label, prev_slc_labels):
	return create_node([label], label, prev_slc_labels)

if __name__ == '__main__':
	#classes = util.classes[1:]
	classes = util.classes

	slc = Slice([create_single_node(classes[i], classes) for i in range(len(classes))], util.nc)
	prev_labels = [node.name for node in slc]
	slices = [slc]

	slc = []
	slc.append(create_node(['cow', 'sheep', 'horse'], 'ungulate', prev_labels))
	slc.append(create_single_node('dog', prev_labels))
	slc.append(create_single_node('cat', prev_labels))
	slc.append(create_single_node('bird', prev_labels))
	slc.append(create_single_node('person', prev_labels))
	slc.append(create_node(['car', 'bus', 'motorbike', 'bicycle', 'train'], 'ground transportation', prev_labels))
	slc.append(create_single_node('boat', prev_labels))
	slc.append(create_single_node('aeroplane', prev_labels))
	slc.append(create_single_node('bottle', prev_labels))
	slc.append(create_single_node('tvmonitor', prev_labels))
	slc.append(create_single_node('potted plant', prev_labels))
	slc.append(create_node(['chair', 'sofa'], 'seating', prev_labels))
	slc.append(create_single_node('diningtable', prev_labels))
	slc = Slice(slc, len(prev_labels))

	prev_labels = [node.name for node in slc]
	slices.append(slc)

	slc = []
	slc.append(create_node(['ungulate', 'dog', 'cat'], 'land animal', prev_labels))
	slc.append(create_single_node('bird', prev_labels))
	slc.append(create_single_node('person', prev_labels))
	slc.append(create_node(['ground transportation', 'boat', 'aeroplane'], 'vehicle', prev_labels))
	slc.append(create_single_node('bottle', prev_labels))
	slc.append(create_single_node('tvmonitor', prev_labels))
	slc.append(create_single_node('potted plant', prev_labels))
	slc.append(create_node(['seating', 'diningtable'], 'furniture', prev_labels))
	slc = Slice(slc, len(prev_labels))

	prev_labels = [node.name for node in slc]
	slices.append(slc)

	slc = []
	slc.append(create_node(['land animal', 'bird', 'person'], 'being', prev_labels))
	slc.append(create_node(['vehicle', 'bottle', 'tvmonitor', 'potted plant', 'furniture'], 'object', prev_labels))
	slc = Slice(slc, len(prev_labels))

	slices.append(slc)

	util.save_slices('slices.pkl', slices)
