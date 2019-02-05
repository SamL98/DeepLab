import numpy as np
from hdf5storage import loadmat
from os.path import join

import sys
imset = sys.argv[1].lower()

ds_path = 'D:/datasets/processed/voc2012'
ds_info = loadmat(join(ds_path, 'ds_info.mat'))
num_img = ds_info['num_'+imset]
class_labels = ds_info['class_labels']

# create mat file mapping each class name to it's children, starts with root -> [foreground, background]
tree = loadmat('decision_tree.mat')
binarized_tree = dict()

for parent, children in tree.items():
    if len(children) <= 2:
        binarized_tree[parent] = children
        continue

    while len(children) > 2:
        binarized_tree[parent] = [children[0], 'not-'+children[0]]
        parent = 'not-'children[0]
        children = children[1:]


subtrees = dict()
for parent, _ in tree.items():
    subtrees[parent] = []

stack = ['root']
curr_path = []

while len(stack) > 0:
    curr = stack[-1]
    children = binarized_tree[curr]
    curr_path.append(curr)

    if len(children) == 0:
        stack.pop()
        curr_path.pop()

        if not curr in class_labels:
            continue

        idx = class_labels.index(curr)
        for node in curr_path:
            subtrees[node].append(idx)

        continue

    for child in children:
        stack.append(child)

gt_path = join(ds_path, 'truth', imset)
gt_fname_fmt = imset+'_%06d_pixeltruth.mat'
gt_mat_name = 'truth_img'

logit_path = join(ds_path, 'Deeplab_Predictions', imset)
logit_fname_fmt = imset+'_%06d_logits.mat'
logit_mat_name = 'logits'

# For each logit vector of each image, get the children of `root` in the tree mat
# and then get the indices of all terminals in their subtrees via the subtrees mat.

for idx in range(1, num_img+1):
    logits = loadmat(join(logit_path, logit_fname_fmt % idx))[logit_mat_name][:]
    logits = logits.reshape(-1, logits.shape[-1])

    gt = loadmat(join(gt_path, gt_fname_fmt % idx))[gt_mat_name][:].flatten()

