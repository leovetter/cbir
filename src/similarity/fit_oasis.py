import pickle
import numpy as np
from oasis import Oasis
import random
import sys
import gzip

def fit(dataset, algo):
    """
    Learn the similarity function with the OASIS algorithm given a dataset
    and the algo used to extract features
    """

    if algo == 'convnet':
        features = pickle.load(open('../../../datasets/Mnist/features/convnet/features_names_and_labels.p', 'rb'))['features']
        labels = pickle.load(open('../../../datasets/Mnist/features/convnet/features_names_and_labels.p', 'rb'))['labels']
    elif algo == 'bag':
        samples = pickle.load(open('../../../datasets/Mnist/features/bag/train_features.p', 'rb'))

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

    if algo == 'convnet':
        pickle.dump(W, open('../convnet/oasis_weights.p', 'wb'))
    elif algo == 'bag':
        pickle.dump(W, open('../bag/oasis_weights.p', 'wb'))

dataset = sys.argv[1]
fit(dataset)
