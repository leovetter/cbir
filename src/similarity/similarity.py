import numpy as np
import pickle

def oasis_similarity(dataset, algo, features_query, db_features, max_rank=100):

    if dataset == 'mnist':

        if algo == 'bag':
            W = pickle.load(open('../../datasets/Mnist/bag_oasis_weights.p', 'rb'))
        elif algo == 'convnet':
            W = pickle.load(open('../../datasets/Mnist/convnet_oasis_weights.p', 'rb'))

        precomp = np.dot(W, db_features.T)
        distance = np.dot(features_query, precomp)

    return np.argsort(np.absolute(distance))[-max_rank:]

def euclidean_similarity(features_query, db_features):

    distances = []
    for features in db_features:

        distance = euclidean(features_query, features)
        distances.append(distance)

    return  np.array(distances).argsort()[:100]
