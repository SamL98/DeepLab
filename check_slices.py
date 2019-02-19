import pickle

with open('slices.pkl', 'rb') as f:
	slices = pickle.load(f)
	
for slice in slices:
	for node in slice:
		print(node.name, node.node_idx, node.terminals)
	print('\n****\n')