import numpy as np
import hdf5storage
from PIL import Image
import os
import tensorflow as tf


HOME_DIR = '/Users/davis.1719/Desktop/Datasets/Processed/VOC2012/'
DECISION_TREE_DIR = HOME_DIR + 'Decision_Tree/'
VOID_LABEL = 255

def get_index(strings, substr):

    for idx, string in enumerate(strings):
        if substr in string:
            break
        elif substr == 'VOID':
            idx = VOID_LABEL
            break

    return idx

def make_mapping(labels, all_node_labels):

    mapping = np.zeros((len(labels),))

    for i in range(len(labels)):
        ind = get_index(all_node_labels, labels[i])
        if ind != VOID_LABEL:
            print('Mapping:', labels[i], ' [{0:d}] -->'.format(i), all_node_labels[ind], '[{0:d}]'.format(ind))
        else:
            print('Mapping:', labels[i], '[{0:d}]'.format(VOID_LABEL), '-->'.format(i), 'VOID', '[{0:d}]'.format(ind))
        mapping[i] = ind

    return mapping


def convert_terminal_img_to_tree_labels(input_to_convert, mapping):

    result = np.zeros(input_to_convert.shape)

    unique_labels = np.unique(input_to_convert)
    for i in unique_labels:
        idx = np.where(input_to_convert == i)
        if i==VOID_LABEL:
            result[idx] = VOID_LABEL
        else:
            result[idx] = mapping[i]

    return result


def convert_terminal_img_to_tree_labels_wrapper():

    # load metadata
    filename = HOME_DIR + 'dataset_info.mat'
    print('Loading:', filename)
    mat_contents = hdf5storage.loadmat(filename)
    num_train = mat_contents['num_train']
    num_val = mat_contents['num_val']
    num_test = mat_contents['num_test']
    num_labels = mat_contents['num_labels']
    class_labels = mat_contents['class_labels']

    # --------------------------------------------------------------

    # set the processing
    input_dir = HOME_DIR + 'Truth/Test/'
    output_dir = input_dir
    num_to_process = num_test
    in_filename_front = 'test_'
    in_filename_back = '_pixeltruth.mat'
    content_to_use = 'truth_img'
    out_filename_back = '_pixeltruth_tree.mat'

    # make directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --------------------------------------------------------------

        # load concept tree
    filename = DECISION_TREE_DIR + 'concept_tree.mat'
    print('loading:', filename)
    mat_contents = hdf5storage.loadmat(filename)

    # terminal labels
    terminal_labels = mat_contents['terminal_labels']
    num_terminal_nodes = len(terminal_labels)
    print('Terminal node labels:')
    for i in range(num_terminal_nodes):
        print(i, terminal_labels[i])

    # nonterminal labels
    nonterminal_labels = mat_contents['nonterminal_labels']
    num_nonterminal_nodes = len(nonterminal_labels)
    print('Nonterminal node labels:')
    for i in range(num_nonterminal_nodes):
        print(i, nonterminal_labels[i])

    # all labels
    all_node_labels = mat_contents['all_node_labels']
    print('All node labels:')
    for i in range(len(all_node_labels)):
        print(i, all_node_labels[i])

    # tree
    tree_parent_rows_child_cols = mat_contents['tree_parent_rows_child_cols']

    # --------------------------------------------------------------

    # get mapping
    mapping = make_mapping(class_labels, all_node_labels)

    # --------------------------------------------------------------

    for i in range(num_to_process):

        filename = input_dir + in_filename_front + '{0:06d}'.format(i+1) + in_filename_back
        print('Reading:', filename)
        mat_contents = hdf5storage.loadmat(filename)
        input_to_convert = mat_contents[content_to_use]
        # other to read in and save?

        # convert it
        result = convert_terminal_img_to_tree_labels(input_to_convert, mapping)

        filename = output_dir + in_filename_front + '{0:06d}'.format(i+1) + out_filename_back
        print('\tSaving:', filename)
        hdf5storage.savemat(filename, {content_to_use: result}) # other to read in and save?


if __name__ == '__main__':

    # run the code
    convert_terminal_img_to_tree_labels_wrapper()


