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
curr_data = None

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

def im_to_b64(arr):
	buff = BytesIO()
	Image.fromarray(arr.astype(np.uint8)).save(buff, format='PNG')
	return str(b64encode(buff.getvalue()))[2:-1]

def arr_to_b64(arr):
	return str(b64encode(arr.astype(np.uint8).tobytes()))[2:-1]

@app.route('/view/<name>/<int:idx>')
def view(name, idx):
	imset = 'test'

	calib_pred = load_calib_pred(imset, idx, name)
	if calib_pred is None:
		return render_template('load_err.html', name='calibrated prediction'), status.HTTP_400_BAD_REQUEST
	masks, conf_maps = calib_pred

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
	dl_pred = dl_pred[:min_h, :min_w]+1
	rgb = rgb[:min_h, :min_w]

	for i, (mask, conf_map) in enumerate(zip(masks, conf_maps)):
		masks[i] = mask[:min_h, :min_w]
		conf_maps[i] = conf_map[:min_h, :min_w]

	global curr_data
	curr_data = {
		'masks': masks,
		'conf_maps': conf_maps
	}

	bg_mask = (1-fg_mask_for(gt)).astype(np.bool)
	dl_pred[bg_mask] = 0

	cmap = voc_colormap()
	max_label = max([mask.max() for mask in masks])
	max_label = max(gt.max(), dl_pred.max(), max_label)

	cmap = np.concatenate((cmap[:max_label], np.expand_dims(cmap[-1], 0)), axis=0)
	gt[gt == 255] = len(cmap)-1

	class_labels = classes
	
	slices = read_slices(join('calib_data', name, 'slices.pkl'))
	for slc in slices[1:]:
		for node in slc:
			class_labels.append(node.name)

	template_info = {
		'idx': idx,
		'conf': conf_thresh,
		'h': min_h,
		'w': min_w,
		'class_labels': class_labels + ['void'],
		'colormap': arr_to_b64(cmap),
		'rgb_str': im_to_b64(rgb),
		'gt_str': arr_to_b64(gt),
		'dl_str': arr_to_b64(dl_pred)
	}
	return render_template('view.html', **template_info)

@app.route('/conf_mask/<float:conf>')
def conf_mask(conf):
	global curr_data
	if not curr_data or (not 'masks' in curr_data) or (not 'conf_maps' in curr_data):
		return None, status.HTTP_400_BAD_REQUEST

	mask = confident_mask(masks, conf_maps, conf)
	return arr_to_b64(mask)

if __name__ == '__main__':
	app.run()