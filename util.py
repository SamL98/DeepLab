import sys
sys.path.insert(0, 'utils')

from node import Node

from utils.ds_utils import *
from utils.loading_utils import *
from utils.mask_utils import *
from utils.eval_utils import *
from utils.slice_utils import *
from utils.multiprocessing_utils import *
from utils.color_utils import *
from utils.viz_utils import *

'''
Expected filesystem:

./
	calib_data/
		<name>/
			slices.pkl
			<node uid>_node_data.mat:
				c_hist
				c_hist
				tot_hist
				acc_hist
				int_ranges

ds_path/
	dataset_info.mat

	RGB/
		<imset>/
			<imset>_<0's>idx_rgb.jpg

	Truth/
		<imset>/
			<imset>_<0's>idx_pixeltruth.mat:
				truth_img

	Deeplab_Prediction/
		<imset>/
			<imset>_<0's>idx_logits.mat:
				logits_img

			<calib name>/
				<imset>_<0's>idx_calib_pred_<conf thresh>.mat
					pred_img
'''

def stdout_writeln(string):
	sys.stdout.write(string + '\n')
	sys.stdout.flush()
