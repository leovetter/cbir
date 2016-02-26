from __future__ import division
import pickle
import glob
import sys
import numpy as np
sys.path.append('../distance/')
from similarity import oasis_similarity, euclidean_similarity
from skimage.io import imread
import random
import cv2
import gzip

def precision(dataset, algo, distance):

    if dataset == 'mnist':

        if algo == 'convnet':

            db_features = pickle.load(open('../../datasets/Mnist/features/convnet/train_features.p', 'rb'))
            labels = pickle.load(open('../../datasets/Mnist/features/convnet/features_names_and_labels.p', 'rb'))['labels']

            if distance == 'oasis':

                all_filenames = glob.glob('../../datasets/Mnist/features/convnet/test_features/*')
                random.shuffle(all_filenames)
                filenames = all_filenames[0:3]

                precisions = []
                recalls = []
                for filename in filenames:

                    label = int(filename.split('/')[-1].split('_')[0])
                    nb_label = (label==labels).sum()
                    query_features = pickle.load(open(filename, 'rb'))

                    query_precisions = []
                    query_recalls = []
                    for max_rank in range(1,1500):
                        id_best_features = oasis_similarity(dataset, algo, query_features, db_features, max_rank)
                        best_labels = np.array(labels)[id_best_features]
                        precision = (best_labels == label).sum()/len(best_labels)
                        recall = (best_labels == label).sum()/nb_label
                        query_precisions.append(precision)
                        query_recalls.append(recall)
                    precisions.append(query_precisions)
                    recalls.append(query_recalls)

                mean_precisions = np.mean(precisions, axis=1)
                pickle.dump(mean_precisions, open('convnet_oasis_mean_precisions.p', 'wb'))
                pickle.dump(recalls, open('convnet_oasis_recalls_at_ranks.p', 'wb'))
                pickle.dump(precisions, open('convnet_oasis_precisions_at_ranks.p', 'wb'))

        elif algo == 'bag':

            samples = np.array(pickle.load(open('../../datasets/Mnist/features/bag/train_features.p', 'rb')))

            db_features = []
            labels = []
            for sample in samples:
                db_features.append(sample['features'])
                labels.append(sample['label'])
                # names.append(sample['name'])
            db_features = np.array(db_features)

            if distance == 'oasis':

                all_filenames = glob.glob('../../datasets/Mnist/images/test/*')
                random.shuffle(all_filenames)
                filenames = all_filenames[0:3]

                f = gzip.open('../../datasets/Mnist/mnist.pkl.gz', 'rb')
                train_set, valid_set, test_set = pickle.load(f)
                f.close()
                X_test, y_test = test_set

                combined = list(zip(X_test, y_test))
                random.shuffle(combined)
                X_test[:], y_test[:] = zip(*combined)
                X_test = X_test[0:3]
                y_test = y_test[0:3]

                precisions = []
                recalls = []
                for img, label in zip(X_test, y_test):

                    print(label)
                    nb_label = (label==labels).sum()
                    print(nb_label)
                    # img = imread(filename)
                    query_features = bag_of_words(img, 64)
                    # label = int(filename.split('/')[-1].split('_')[0])

                    query_precisions = []
                    query_recalls = []
                    for max_rank in range(1,1500):
                        id_best_features = oasis_similarity(dataset, algo, query_features, db_features, max_rank)
                        best_labels = np.array(labels)[id_best_features]
                        precision = (best_labels == label).sum()/len(best_labels)
                        recall = (best_labels == label).sum()/nb_label
                        query_precisions.append(precision)
                        query_recalls.append(recall)
                    precisions.append(query_precisions)
                    recalls.append(query_recalls)

                mean_precisions = np.mean(precisions, axis=1)
                pickle.dump(mean_precisions, open('bag_oasis_mean_precisions.p', 'wb'))
                pickle.dump(recalls, open('bag_oasis_recalls_at_ranks.p', 'wb'))
                pickle.dump(precisions, open('bag_oasis_precisions_at_ranks.p', 'wb'))


def bag_of_words(img, nb_bins):

    img = np.reshape(img, (28,28)) * 255
    img = img.astype('uint8')

    sift = cv2.SIFT(contrastThreshold=0.01, edgeThreshold=50, sigma=0.8)
    kp, des = sift.detectAndCompute(img,None)

    model = pickle.load(open('../features/bag/output/mnist/64-cluster/train_kmeans_model.npy', 'rb'))

    features = model.predict(des)
    histo = np.bincount(features, minlength=nb_bins)

    return np.array(histo)

dataset = sys.argv[1]
algo = sys.argv[2]
distance = sys.argv[3]
precision(dataset, algo, distance)
