import multiprocessing as mp
from contextlib import contextmanager

import os
import signal

from os.path import isfile 
import numpy as np

@contextmanager
def poolcontext(num_proc):
    pool = mp.Pool(num_proc)
    yield pool
    pool.terminate()

def get_param_batches(slices, args):
	idx_ordering = None
	idx_ordering_fname = args.imset.lower() + '_ordered.txt'

	if not isfile(idx_ordering_fname):
		from order_by_num_fg import order_imset_by_num_fg
		idx_ordering = order_imset_by_num_fg(args.imset, save=True)
	else:
		with open(idx_ordering_fname) as f:
			idx_ordering = [int(idx) for idx in f.read().split('\n')]

	idx_ordering = np.array(idx_ordering)
	if args.test:
		idx_ordering = idx_ordering[:2*args.num_proc]
		
	param_batches = []

	for procno in range(args.num_proc):
		idx_batch = idx_ordering[procno::args.num_proc]
		param_batches.append((idx_batch, slices.copy(), args))

	return param_batches

def kill_children(proc_slices):
	for proc_slc in proc_slices:
		os.kill(proc_slc.pid, signal.SIGTERM)
