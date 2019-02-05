import os
from os.path import join

from PIL import Image
import numpy as np

root_dir = join('D:', 'datasets', 'original', 'vocdevkit', 'voc2012')
src_dir = join(root_dir, 'segmentationclass')
dst_dir = join(root_dir, 'SegmentationClassRaw')

if not os.path.isdir(dst_dir):
	os.mkdir(dst_dir)

for f in os.listdir(src_dir):
	arr = np.array(Image.open(join(src_dir, f)))
	Image.fromarray(arr).save(join(dst_dir, f))
