import util
import numpy as np

imset = 'val'
nb = 20
res = 1./nb
batch_size = 500000
T = 0.8

valib_uncalib_corr_hist = np.zeros((nb), dtype=np.float64)
valib_uncalib_count_hist = np.zeros((nb), dtype=np.float64)

lgts = np.zeros((batch_size, util.nc), dtype=util.DTYPES[util.LOGITS])
gt = np.zeros((batch_size), dtype=util.DTYPES[util.GT])

def accum_hists(lgts, term_preds, gt, corr_hist, count_hist, T):
	sm = util.sm_of_logits(lgts, T=T)

	binno = np.floor(sm/res).astype(np.uint8)
	binno = np.minimum(binno, nb-1)

	for bn in np.unique(binno):
		bin_mask = binno[binno == bn]
		valib_uncalib_corr_hist[binno] += (term_preds[bin_mask] == gt[bin_mask]).sum()
		valib_uncalib_count_hist[binno] += bin_mask.sum()

def prec_hists_for_imset(imset):
	uncalib_corr_hist = np.zeros((nb), dtype=np.uint64)
	uncalib_count_hist = np.zeros((nb), dtype=np.uint64)

	calib_corr_hist = np.zeros((nb), dtype=np.uint64)
	calib_count_hist = np.zeros((nb), dtype=np.uint64)

	for chunkno in range(8):
		done = False

		while not done:
			done, num_pix = util.unserialize_examples_for_calib(imset, batch_size, chunkno, lgts, gt) 	
			term_preds = np.argmax(lgts, -1)
			gt -= 1

			if done:
				lgts = lgts[:num_pix]
				gt = gt[:num_pix]
				term_preds = term_preds[:num_pix]

			accum_hists(lgts, term_preds, gt, uncalib_corr_hist, uncalib_count_hist, 1)
			accum_hists(lgts, term_preds, gt, calib_corr_hist, calib_count_hist, T)

	uncalib_prec_hist = uncalib_corr_hist.astype(np.float64) / np.maximum(1e-7, uncalib_count_hist.astype(np.float64))
	calib_prec_hist = calib_corr_hist.astype(np.float64) / np.maximum(1e-7, calib_count_hist.astype(np.float64))

	return uncalib_prec_hist, calib_prec_hist

val_unc_phist, val_c_phist = prec_hists_for_imset('val')
test_unc_phist, test_c_phist = prec_hists_for_imset('test')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2)

ax[0,0].set_title('Val Uncalibrated')
ax[0,0].scatter(range(nb), val_unc_phist)
ax[0,0].line(range(nb), np.linspace(0, 1, num=nb), linestyle='--')

ax[0,1].set_title('Val Calibrated, T = %f' % T)
ax[0,1].scatter(range(nb), val_c_phist)
ax[0,1].line(range(nb), np.linspace(0, 1, num=nb), linestyle='--')

ax[1,0].set_title('Test Uncalibrated')
ax[1,0].scatter(range(nb), test_unc_phist)
ax[1,0].line(range(nb), np.linspace(0, 1, num=nb), linestyle='--')

ax[1,1].set_title('Test Calibrated, T = %f' % T)
ax[1,1].scatter(range(nb), test_c_phist)
ax[1,1].line(range(nb), np.linspace(0, 1, num=nb), linestyle='--')

fig.savefig('images/temp_scaling_%f.png' % T, bbox_inches='tight')
plt.show()
