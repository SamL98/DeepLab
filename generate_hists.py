from util import *

import multiprocessing as mp
from contextlib import contextmanager

@contextmanager
def poolcontext(num_proc):
	pool = mp.Pool(num_proc)
	yield pool
	pool.terminate()

def get_hist(slc):
	for node in slc:
		node.generate_acc_hist(10, equa=False)

if __name__ == '__main__':
	slices = read_slices('slices.pkl')

	num_proc = len(slices)

	with poolcontext(num_proc) as p:
		proc_slices = p.map(get_hist, slices)

	save_slices('slices.pkl', slices)
