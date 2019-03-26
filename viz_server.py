from util import *
from os.path import join, isdir
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_api import status
from base64 import b64encode
import re
from PIL import Image
from io import BytesIO

app = Flask(__name__, static_folder='static', static_url_path='/static')

curr_masks = None

@app.route('/')
def main():
	base_dir = join(ds_path, 'deeplab_prediction', 'test')
	valid_names = os.listdir(base_dir)
	valid_names = list(filter(lambda f: isdir(join(base_dir, f)) and (not f == 'dl_pred'), valid_names))
	template_info = {
		'valid_names': valid_names,
		'num_img': num_img_for('test')
	}
	return render_template('index.html', **template_info)

@app.route('/valid_confs/<name>/<int:idx>')
def valid_confs(name, idx):
	p = re.compile('test_(0+)(1)_calib_pred_0.(\d+).mat')

	fnames = os.listdir(join(ds_path, 'deeplab_prediction', 'test', name))
	valid_fnames = filter(lambda f: p.match(f), fnames)
	valid_confs = list(set(map(lambda f: float(f[f.rindex('_')+1:f.rindex('.')]), valid_fnames)))

	return jsonify(valid_confs)

def im_to_b64(arr):
	buff = BytesIO()
	Image.fromarray(arr.astype(np.uint8)).save(buff, format='PNG')
	return str(b64encode(buff.getvalue()))[2:-1]

def arr_to_b64(arr):
	return str(b64encode(arr.astype(np.uint8).tobytes()))[2:-1]

@app.route('/view/<name>/<int:idx>')
def view(name, idx):
	conf_thresh = float(request.args.get('conf'))
	imset = 'test'

	calib_pred = load_calib_pred(imset, idx, conf_thresh, name)
	if calib_pred is None:
		return render_template('load_err.html', name='calibrated prediction'), status.HTTP_400_BAD_REQUEST

	dl_pred = load_dl_pred(imset, idx)
	if dl_pred is None:
		return render_template('load_err.html', name='DeepLab prediction'), status.HTTP_400_BAD_REQUEST

	gt = load_gt(imset, idx, reshape=False)
	if gt is None:
		return render_template('load_err.html', name='ground truth mask'), status.HTTP_400_BAD_REQUEST

	rgb = load_rgb(imset, idx)
	if dl_pred is None:
		return render_template('load_err.html', name='rgb image'), status.HTTP_400_BAD_REQUEST

	min_h = min(len(gt), len(dl_pred))
	min_w = min(gt.shape[1], dl_pred.shape[1])

	gt = gt[:min_h, :min_w]
	dl_pred = dl_pred[:min_h, :min_w]
	rgb = rgb[:min_h, :min_w]
	calib_pred = calib_pred[:min_h, :min_w]

	bg_mask = (1-fg_mask_for(gt)).astype(np.bool)
	dl_pred[bg_mask] = 0

	cmap = voc_colormap()
	max_label = max(gt.max(), dl_pred.max(), calib_pred.max())
	cmap = np.concatenate((cmap[:max_label], np.expand_dims(cmap[-1], 0)), axis=0)
	gt[gt==255] = len(cmap)-1

	diff_mask = calib_pred != dl_pred
	where_diff = np.where(diff_mask)[0].astype(np.uint8)
	diff_labs = calib_pred[diff_mask].astype(np.uint8)
	calib_delta = list(map(list, zip(where_diff, diff_labs)))

	global curr_masks
	curr_masks = {
		'calib_pred': calib_pred,
		'dl_pred': dl_pred,
		'gt': gt
	}

	template_info = {
		'idx': idx,
		'conf': conf_thresh,
		'h': min_h,
		'w': min_w,
		'class_labels': classes + ['void'],
		'colormap': arr_to_b64(cmap),
		'rgb_str': im_to_b64(rgb),
		'gt_str': arr_to_b64(gt),
		'dl_str': arr_to_b64(dl_pred),
		'calib_delta': calib_delta
	}
	return render_template('view.html', **template_info)

if __name__ == '__main__':
	app.run()