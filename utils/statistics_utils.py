import numpy as np
from scipy.stats import norm

def conf_ints(acc_hist, count_hist, alpha):
	mask = count_hist > 0
	ranges = np.zeros_like(acc_hist)

	p, n = acc_hist[mask], count_hist[mask]
	
	z = norm.ppf(1 - alpha/2)
	pq = p * (1 - p)
	zn = z**2 / (4*n)
	
	if (pq < 0).sum() > 0:
		sys.stdout.write('acc_hist: ' + acc_hist.__repr__() + '\n')
		sys.stdout.flush()
		exit()

	conf_range = z * np.sqrt((pq + zn) / n) / (1 + zn*4)
	new_p = (p + zn*2) / (1 + zn*4)

	conf_range = np.clip(conf_range, 0, new_p)
	conf_range = np.minimum(conf_range, 1-new_p)

	acc_hist[mask] = new_p
	ranges[mask] = conf_range

	return acc_hist, ranges

def parzen_estimate(confs, bins, sigma):
	parzen = np.zeros_like(bins)
	for i, bn in enumerate(bins):
		z = (confs - bn) / sigma
		parzen[i] = (np.exp(-z**2 / 2)).sum()
	return parzen
