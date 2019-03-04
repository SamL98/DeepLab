from util import *

import multiprocessing as mp
from contextlib import contextmanager

@contextmanager
def poolcontext(num_proc):
	pool = mp.Pool(num_proc)
	yield pool
	pool.terminate()

def get_hist(slc):
	slc_len = len(slc)
	for node in slc:
		node.generate_acc_hist(10, slc_len, equa=False)
	return slc

if __name__ == '__main__':
	slices = read_slices('slices.pkl', reset=False)

	num_proc = len(slices)

	with poolcontext(num_proc) as p:
		slices = p.map(get_hist, slices)

	save_slices('slices.pkl', slices)
