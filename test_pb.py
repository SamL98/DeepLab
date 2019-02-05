import tensorflow as tf
from skimage.io import imread
import numpy as np

PATH_TO_DEEPLAB_GRAPH = 'model_trained/deeplabv3_pascal_train_aug/frozen_inference_graph.pb'

tf.reset_default_graph()

with tf.gfile.GFile(PATH_TO_DEEPLAB_GRAPH, 'rb') as f:
    graph_def = tf.GraphDef.FromString(f.read())	

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')

with tf.Session(graph=graph) as sess:
    var = graph.get_tensor_by_name('concat:0')
    print(var)
    # testIm = imread('../refinenet/datasets/PascalContext/RGB/Train/train_%06d_img.png' % 1)
    # testIm = np.expand_dims(testIm, axis=0)
    # tens = sess.run(var, feed_dict={'ImageTensor:0': testIm})
    # print(tens.shape)