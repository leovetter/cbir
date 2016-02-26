from __future__ import division
import sys
sys.path.append('../')
from load_data import load_pascal_set
import cv2
import sys
from sklearn.cluster import KMeans
import numpy as np
import pickle
from skimage.viewer import ImageViewer
from sklearn.naive_bayes import MultinomialNB
import gzip
from sklearn.svm import SVC
import random
from collections import defaultdict

def train_features():

    datasets = load_pascal_set('train', False, 'aeroplane', 'bicycle', 'car', 'bird', 'boat', 'person')
    train_set_x, train_set_y, train_images = datasets[0]
    val_set_x, val_set_y, val_images = datasets[1]

    all_descriptors = None
    print('extract descriptors with SIFT')
    desc_by_img = []
    for img in train_set_x:

        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(gray,None)

        desc_by_img.append(des)

        if all_descriptors is None:
            all_descriptors = des
        else:
            all_descriptors = np.concatenate([all_descriptors, des])

    print('Find vocabulary with K-means clustering')
    #pickle.dump(all_descriptors, open('all_descriptors.npy', 'wb'))
    #pickle.dump(desc_by_img, open('desc_by_img.npy', 'wb'))
    for nb_cluster in [8]:

        model = KMeans(n_clusters=nb_cluster)
        model.fit(all_descriptors)

        print('Compute histos')
        features = []
        for descriptors, img_name, label in zip(desc_by_img, train_images, train_set_y):

            feats = model.predict(descriptors)
            histo = np.bincount(feats, minlength=nb_cluster)
            feat_name_and_label = {}
            feat_name_and_label['features'] = histo
            feat_name_and_label['name'] = img_name
            feat_name_and_label['label'] = label
            features.append(feat_name_and_label)

            f_features = open('output/pascal/'+str(nb_cluster)+'-cluster/train_features.npy', 'wb')
            pickle.dump(features, f_features)
            f_features.close()
            f_model = open('output/pascal/'+str(nb_cluster)+'-cluster/train_kmeans_model.npy', 'wb')
            pickle.dump(model, f_model)
            f_model.close()

def svm_classifier(nb_cluster, dataset_name):

    features_names_and_labels = pickle.load(open('output/'+dataset_name+'/'+str(nb_cluster)+'-cluster/train_features.npy', 'rb'))
    labels = [ triplet['label'] for triplet in features_names_and_labels ]

    features = []
    for triplet in features_names_and_labels:
        features.append(triplet['features'])
    features = np.array(features)

    clf = SVC()
    clf.fit(features, labels)

    pickle.dump(clf, open('output/'+dataset_name+'/'+str(nb_cluster)+'-cluster/train_svm_classifier.npy', 'wb'))

def bayes_classifier(nb_cluster):

    features_names_and_labels = pickle.load(open('output/pascal/'+str(nb_cluster)+'-cluster/train_features.npy', 'rb'))
    features = [ triplet['features'] for triplet in features_names_and_labels ]
    labels = [ triplet['label'] for triplet in features_names_and_labels ]

    clf = MultinomialNB()
    print('fit')
    clf.fit(features, labels)

    pickle.dump(clf, open('output/pascal/'+str(nb_cluster)+'-cluster/train_bayes_classifier.npy', 'wb'))

def mnist_val_features(nb_cluster):

    f = gzip.open('../../../datasets/Mnist/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    X_test, y_test = test_set

    combined = list(zip(X_valid, y_valid))
    random.shuffle(combined)
    X_valid[:], y_valid[:] = zip(*combined)

    X_valid = X_valid[:5000]
    y_valid = y_valid[:5000]

    model = pickle.load(open('output/mnist/'+str(nb_cluster)+'-cluster/train_kmeans_model.npy', 'rb'))

    val_features = []
    for img, label in zip(X_valid, y_valid):

        img = np.reshape(img, (28,28)) * 255
        img = img.astype('uint8')
        # gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT(contrastThreshold=0.01, edgeThreshold=50, sigma=0.8)
        kp, des = sift.detectAndCompute(img,None)

        features = model.predict(des)
        histo = np.bincount(features, minlength=nb_cluster)

        feats_label = {}
        feats_label['features'] = histo
        # feat_name_and_label['name'] = img_name
        feats_label['label'] = label
        val_features.append(feats_label)

    pickle.dump(val_features, open('output/mnist/'+str(nb_cluster)+'-cluster/val_features.npy', 'wb'))

def pascal_val_features(nb_cluster):

    datasets = load_pascal_set('train', False, 'aeroplane', 'bicycle', 'car', 'bird', 'boat', 'person')
    val_set_x, val_set_y, val_images = datasets[1]

    model = pickle.load(open('output/pascal/'+str(nb_cluster)+'-cluster/train_kmeans_model.npy', 'rb'))

    val_features = []
    for img in val_set_x:
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(gray,None)

        features = model.predict(des)
        histo = np.bincount(features, minlength=nb_cluster)
        val_features.append(histo)

    pickle.dump(val_features, open('output/pascal/'+str(nb_cluster)+'-cluster/val_features.npy', 'wb'))

def val_bag_of_words_prediction():

    datasets = load_pascal_set('train', False, 'aeroplane', 'bicycle', 'car', 'bird', 'boat', 'person')
    _, val_set_y, _ = datasets[1]

    accuracies = []
    nb_clusters = []
    for nb_cluster in [32]:

        val_features = pickle.load(open('output/pascal/'+str(nb_cluster)+'-cluster/val_features.npy', 'rb'))
        clf = pickle.load(open('output/pascal/'+str(nb_cluster)+'-cluster/train_svm_classifier.npy', 'rb'))

        y_pred = clf.predict(val_features)
        accuracy = (y_pred == val_set_y).sum()/len(y_pred)
        print("Accuracy of the model for nb clusters of %f is : %f" % (nb_cluster, accuracy))
        accuracies.append(accuracy)
        nb_clusters.append(nb_cluster)
        pickle.dump(accuracies, open('output/pascal/accuracies.npy', 'wb'))
        pickle.dump(nb_clusters, open('output/pascal/nb_clusters.npy', 'wb'))

def mnist_train_feature():

    f = gzip.open('../../../datasets/Mnist/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    X_test, y_test = test_set

    nb_labels = defaultdict(int)
    names = []
    for image, label in zip(X_train, y_train):

        nb_labels[label] += 1

        img = np.reshape(image, (28,28))
        name = 'images/train/'+str(label)+'_'+str(nb_labels[label])+'.jpeg'
        names.append(name)

    combined = list(zip(X_train, y_train, names))
    random.shuffle(combined)
    X_train[:], y_train[:], names[:] = zip(*combined)

    X_train = X_train[:5000]
    y_train = y_train[:5000]
    names = names[:5000]

    all_descriptors = None
    print('extract descriptors with SIFT')
    desc_by_img = []
    for img in X_train:

        img = np.reshape(img, (28,28)) * 255
        img = img.astype('uint8')
        # gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT(contrastThreshold=0.01, edgeThreshold=50, sigma=0.8)
        kp, des = sift.detectAndCompute(img,None)
        desc_by_img.append(des)

        if all_descriptors is None:
            all_descriptors = des
        else:
            all_descriptors = np.concatenate([all_descriptors, des])

    print('Find vocabulary with K-means clustering')
    print(all_descriptors.shape)
    for nb_cluster in [64]:

        model = KMeans(n_clusters=nb_cluster)
        model.fit(all_descriptors)

        print('Compute histos')
        features = []
        for descriptors, label, name in zip(desc_by_img, y_train, names):

            feats = model.predict(descriptors)
            histo = np.bincount(feats, minlength=nb_cluster)
            feat_name_and_label = {}
            feat_name_and_label['features'] = histo
            feat_name_and_label['name'] = name
            feat_name_and_label['label'] = label
            features.append(feat_name_and_label)

            pickle.dump(features, open('output/mnist/'+str(nb_cluster)+'-cluster/train_features.npy', 'wb'))
            pickle.dump(model, open('output/mnist/'+str(nb_cluster)+'-cluster/train_kmeans_model.npy', 'wb'))

def mnist_prediction():

    accuracies = []
    nb_clusters = []
    for nb_cluster in [8, 20, 32, 52, 64]:

        val_features_labels = pickle.load(open('output/mnist/'+str(nb_cluster)+'-cluster/val_features.npy', 'rb'))
        val_set_y = [doublet['label'] for doublet in val_features_labels]

        val_features = []
        for doublet in val_features_labels:
            val_features.append(doublet['features'])
        val_features = np.array(val_features)

        clf = pickle.load(open('output/mnist/'+str(nb_cluster)+'-cluster/train_svm_classifier.npy', 'rb'))

        y_pred = clf.predict(val_features)

        accuracy = (y_pred == val_set_y).sum()/len(y_pred)
        print("Accuracy of the model for nb clusters of %f is : %f" % (nb_cluster, accuracy))
        accuracies.append(accuracy)
        nb_clusters.append(nb_cluster)
        pickle.dump(accuracies, open('output/mnist/accuracies.npy', 'wb'))
        pickle.dump(nb_clusters, open('output/mnist/nb_clusters.npy', 'wb'))

# train_features()
# svm_classifier(8, 'mnist')
# svm_classifier(32)
# val_features(8)
#val_bag_of_words_prediction()

mnist_train_feature()
# svm_classifier(20, 'mnist')
# svm_classifier(32, 'mnist')
# svm_classifier(40, 'mnist')
# svm_classifier(52, 'mnist')
# svm_classifier(64, 'mnist')
# mnist_val_features(20)
# mnist_val_features(32)
# mnist_val_features(40)
# mnist_val_features(52)
# mnist_val_features(64)
# mnist_prediction()

# bag_of_words_classifier(8)
#bag_of_words_classifier(20)
# bag_of_words_classifier(20)
# bag_of_words_classifier(40)

#val_features(40)
# val_features(20)
# val_features(40)
