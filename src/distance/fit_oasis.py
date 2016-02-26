import pickle
import numpy as np
from oasis import Oasis
import random
import sys
import gzip

def bag_similarity():

    features_names_and_labels = pickle.load(open('../algos/bag/output/pascal/32-cluster/train_features.npy', 'rb'))
    labels = [ triplet['label'] for triplet in features_names_and_labels ]
    _, labels = np.unique(labels, return_inverse=True)

    features = []
    for triplet in features_names_and_labels:
        features.append(triplet['features'])
    features = np.array(features)

    print(features.shape)

    combined = list(zip(features, labels))
    random.shuffle(combined)
    features[:], labels[:] = zip(*combined)

    model = Oasis(n_iter=100000, do_psd=True, psd_every=3,
                  save_path="/tmp/gwtaylor/oasis_test").fit(features, labels,
                                                            verbose=True)
    W = model._weights.view()
    W.shape = (np.int(np.sqrt(W.shape[0])), np.int(np.sqrt(W.shape[0])))

    pickle.dump(W, open('../algos/bag/output/pascal/32-cluster/oasis_weights.npy', 'wb'))

def fit(dataset):

    # f = gzip.open('../../../datasets/Mnist/mnist.pkl.gz', 'rb')
    # train_set, _, _ = pickle.load(f)
    # f.close()
    #
    # _, labels = train_set

    # features = pickle.load(open('../../../datasets/Mnist/convnet_train_features.p', 'rb'))
    samples = pickle.load(open('../../../datasets/Mnist/bag_train_features.p', 'rb'))

    features = []
    labels = []
    for sample in samples:
        features.append(sample['features'])
        labels.append(sample['label'])
    features = np.array(features)
    labels = np.array(labels)

    model = Oasis(n_iter=100000, do_psd=True, psd_every=3,
                  save_path="/tmp/gwtaylor/oasis_test").fit(features, labels,
                                                            verbose=True)
    W = model._weights.view()

    W.shape = (np.int(np.sqrt(W.shape[0])), np.int(np.sqrt(W.shape[0])))

    # pickle.dump(W, open('../convnet/oasis_weights.p', 'wb'))
    pickle.dump(W, open('../bag/oasis_weights.p', 'wb'))

dataset = sys.argv[1]
fit(dataset)
