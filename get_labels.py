from hdf5storage import loadmat

labels = loadmat('D:/datasets/processed/voc2012/dataset_info.mat')['class_labels']
print(labels)
