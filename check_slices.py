import pickle
from collections import namedtuple
Cluster = namedtuple('Cluster', 'name cluster_idx terminals corr_hist count_hist acc_hist')

with open('slices.pkl', 'rb') as f:
	slices = pickle.load(f)
	
for slice in slices:
	for cluster in slice:
		print(cluster.name, cluster.cluster_idx, cluster.terminals)
	print('\n****\n')