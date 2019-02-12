import os
import tensorflow as tf

ckpt = tf.train.get_checkpoint_state('E:/lerner/deeplab/cache_data/model-pc-1541640734/train')
inpt_ckpt = ckpt.model_checkpoint_path

output_graph = 'E:/lerner/deeplab/cache_Data/model-pc-1541640734/frozen_inference_graph-2.pb'

with tf.Session(graph=tf.Graph()) as sess:
	saver = tf.train.import_meta_graph(inpt_ckpt+'.meta', clear_devices=True)
	saver.restore(sess, inpt_ckpt)

	names = [op.name for op in tf.get_default_graph().get_operations()]
	with open('E:/lerner/deeplab/varnames.txt', 'w') as f:
		f.write('\n'.join(names))
	exit()	

	var = tf.get_default_graph().get_tensor_by_name('image')
	print(var)
	exit()
 
	output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
																	tf.get_default_graph().as_graph_def(),
																	['image', 'decoder/concat'])

	with tf.gfile.GFile(output_graph, 'wb') as f:
		f.write(output_graph_def.SerializeToString())

