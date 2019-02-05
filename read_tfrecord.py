import tensorflow as tf

with tf.Session() as sess:
	feat = {
		'image/encoded': tf.FixedLenFeature([], tf.string),
		'image/segmentation/class/encoded': tf.FixedLenFeature([], tf.string),
		'image/height': tf.FixedLenFeature([], tf.int64),
		'image/width': tf.FixedLenFeature([], tf.int64)
	}
	queue = tf.train.string_input_producer(['datasets/voc2012/val-00000-of-00004.tfrecord'], num_epochs=1)
	reader = tf.TFRecordReader()
	_, serialized_ex = reader.read(queue)
	features = tf.parse_single_example(serialized_ex, features=feat)

	w, h = tf.cast(features['image/width'], tf.int32), tf.cast(features['image/height'], tf.int32)
	print(h, w)

	image = tf.image.decode_png(features['image/encoded'], channels=3)
	#image = tf.decode_raw(features['image/encoded'], tf.float32)
	image = tf.reshape(tf.stack(image), [1, h, w, 3])
	#image.set_shape([1, h, w, 3])

	labelmap = tf.image.decode_png(features['image/segmentation/class/encoded'], channels=1)
	labelmap = tf.reshape(tf.stack(labelmap), [1, h, w, 1])
	#labelmap.set_shape([1, h, w, 1])

	#images, labelmaps = tf.train.shuffle_batch([image, labelmap], batch_size=1, capacity=30, num_threads=1, min_after_dequeue=10)

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	
	#im, mask = sess.run([images, labelmaps])
	im, mask = sess.run([image, labelmap])

coord.request_stop()
coord.join(threads)
sess.close()

im, mask = im[0,...], mask[0,...]
if len(mask.shape) == 3:
	mask = mask[:,:,0]
print(im.shape, mask.shape)

import matplotlib.pyplot as plt
_, a = plt.subplots(1, 2)
a[0].imshow(im)
a[1].imshow(mask, 'jet')
plt.show()
