from PIL import Image
import numpy as np

im = Image.open('D:\\Datasets\\Original\\VOCdevkit\\VOC2012\\SegmentationClass\\2007_000129.png')
arr = np.array(im)
print(np.unique(arr))
print(arr.shape)

im = im.convert('RGB')
arr = np.array(im)
print(np.unique(arr))
print(arr.shape)
