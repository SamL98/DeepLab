from util import *
from node import *
import numpy as np

np.random.seed(1234)

m = 1000
n_iter = 10

num_tot = 0
num_corr = 0

test_node = Node('test', -1, [])

for i in range(n_iter):
	confs = np.random.uniform(0, 1, (m))
	correct_mask = np.random.choice([0, 1], (m), replace=True).astype(np.bool)

	test_node.accum_scores(confs, correct_mask, 100, 0.1)

	num_tot += len(confs)
	num_corr += correct_mask.sum()

	node_num_tot = test_node.tot_hist.sum()
	tot_perc_diff = abs(num_tot - node_num_tot) / float(num_tot)

	assert tot_perc_diff < 0.01, 'Node not consistent with num_tot after %d iterations: %d vs %d' % (i+1, node_num_tot, num_tot)

	node_num_corr = test_node.c_hist.sum()
	c_perc_diff = abs(num_corr - node_num_corr) / float(num_corr)

	assert c_perc_diff < 0.01, 'Node not consistent with num_corr after %d iterations: %d vs %d' % (i+1, node_num_corr, num_corr)