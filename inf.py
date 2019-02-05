import tensorflow as tf
from skimage.io import imread
import numpy as np
from hdf5storage import savemat

from os.path import join

GRAPH_PATH = 'E:/lerner/deeplab/model_trained/deeplabv3_pascal_train_aug/frozen_inference_graph.pb' 
DS_PATH = 'D:/datasets/processed/voc2012' 
RGB_PATH = join(DS_PATH, 'rgb', 'val')

INPT_FMT = 'val_%06d_img.png' 
im = imread(join(RGB_PATH, INPT_FMT % 428)) 
h, w, _ = im.shape

CROP_SIZE = 513
pad_h = max(0, CROP_SIZE-h)
pad_w = max(0, CROP_SIZE-w)

im = np.pad(
	im,
	[(0, pad_h), (0, pad_w), (0, 0)],
	'constant',
	constant_values=128)
im = np.expand_dims(im, axis=0)

tf.reset_default_graph()

with tf.gfile.GFile(GRAPH_PATH, 'rb') as f: 
	graph_def = tf.GraphDef.FromString(f.read())

with tf.Graph().as_default() as graph:
	tf.import_graph_def(graph_def, name='') 

with tf.Session(graph=graph) as sess:
	labels = sess.run(graph.get_tensor_by_name('SemanticPredictions:0'),
			feed_dict={'ImageTensor:0': im})
	labels = np.squeeze(labels)[:h, :w]
	savemat('test.mat', {'pred': labels})
