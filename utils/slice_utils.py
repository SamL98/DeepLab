import pickle
import numpy as np

def read_slices(fname):
	with open(fname, 'rb') as f:
		slices = pickle.load(f)
	return slices

def save_slices(fname, slices):
	with open(fname, 'wb') as f:
		pickle.dump(slices, f)
