from __future__ import division
import sys
sys.path.append('../')
from load_data import load_pascal_set, load_paris_set
import numpy as np
import random
from deepnet import ConvNet
import gzip
import pickle
from collections import defaultdict
from skimage.viewer import ImageViewer
from sklearn.linear_model import Perceptron
from sklearn import svm

def check(train_set_x, train_set_y, val_set_x, val_set_y):

    viewer = ImageViewer(np.transpose(val_set_x[0], (1,2,0)))
    print(val_set_y[0])
    viewer.show()

    viewer = ImageViewer(np.transpose(val_set_x[10], (1,2,0)))
    print(val_set_y[10])
    viewer.show()

    viewer = ImageViewer(np.transpose(val_set_x[123], (1,2,0)))
    print(val_set_y[123])
    viewer.show()


    viewer = ImageViewer(np.transpose(val_set_x[37], (1,2,0)))
    print(val_set_y[37])
    viewer.show()

    viewer = ImageViewer(np.transpose(val_set_x[176], (1,2,0)))
    print(val_set_y[176])
    viewer.show()

    viewer = ImageViewer(np.transpose(val_set_x[331], (1,2,0)))
    print(val_set_y[331])
    viewer.show()

    viewer = ImageViewer(np.transpose(val_set_x[376], (1,2,0)))
    print(val_set_y[376])
    viewer.show()

    viewer = ImageViewer(np.transpose(val_set_x[234], (1,2,0)))
    print(val_set_y[234])
    viewer.show()

    viewer = ImageViewer(np.transpose(val_set_x[198], (1,2,0)))
    print(val_set_y[198])
    viewer.show()

    viewer = ImageViewer(np.transpose(val_set_x[200], (1,2,0)))
    print(val_set_y[200])
    viewer.show()

def train_convnet(dataset):

    if dataset == 'mnist':
        train_mnist()
    elif dataset == 'paris':
        train_paris()
    elif dataset == 'pascal':
        train_pascal()
    else:
        print('give correct dataset name')

def train_pascal():

    datasets = load_pascal_set('train', True, 'aeroplane', 'bicycle', 'car', 'bird', 'boat', 'person')
    train_set_x, train_set_y, train_images = datasets[0]
    val_set_x, val_set_y, val_images = datasets[1]

    train_set_x = np.array(train_set_x)
    val_set_x = np.array(val_set_x)

    _,train_set_y = np.unique(train_set_y, return_inverse=True)
    _,val_set_y = np.unique(val_set_y, return_inverse=True)

    combined = list(zip(train_set_x, train_set_y))
    random.shuffle(combined)
    train_set_x[:], train_set_y[:] = zip(*combined)

    combined = list(zip(val_set_x, val_set_y))
    random.shuffle(combined)
    val_set_x[:], val_set_y[:] = zip(*combined)

    train_set_x = np.transpose(train_set_x, (0,3,1,2))
    val_set_x = np.transpose(val_set_x, (0,3,1,2))

def train_mnist():

f = gzip.open('../../../datasets/Mnist/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f)
f.close()

X_train, train_set_y = train_set
X_valid, val_set_y = valid_set
X_test, y_test = test_set

    y_test = np.array(y_test)

    train_set_x = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
    val_set_x = np.reshape(X_valid, (X_valid.shape[0], 1, 28, 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

    print('start training')
    convnet = ConvNet(train_set_x, train_set_y, val_set_x, val_set_y, 50)
    convnet.fit(curves=True, max_epoch=20)

def train_paris():

    datasets = load_paris_set()
    train_set_x, train_set_y, train_images = datasets[0]
    val_set_x, val_set_y, val_images = datasets[1]

    # print(train_set_x.shape)
    # print(val_set_x.shape)
    print(len(np.unique(train_set_y)))
    train_set_x = np.reshape(train_set_x, (377,45000))
    val_set_x = np.reshape(val_set_x, (270,45000))

    # clf = Perceptron(n_iter=50)
    clf = svm.SVC()
    clf.fit(train_set_x, train_set_y)
    y_pred = clf.predict(val_set_x)

    print("Accuracy of the model is : %f" % ((y_pred == val_set_y).sum()/len(val_set_y)))
    # convnet = ConvNet(train_set_x, train_set_y, val_set_x, val_set_y, 100)
    # convnet.fit(curves=False, max_epoch=100)

def train_features():

    f = gzip.open('../../../datasets/Mnist/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    X_train, train_set_y = train_set

    nb_labels = defaultdict(int)
    names = []
    for image, label in zip(X_train, train_set_y):

        nb_labels[label] += 1

        img = np.reshape(image, (28,28))
        name = 'images/train/'+str(label)+'_'+str(nb_labels[label])+'.jpeg'
        names.append(name)


    train_set_x = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
    convnet = pickle.load(open('training_outputs/best_model.p', 'rb'))
    features = convnet.features(train_set_x)
    print(features.shape)

    features_names_and_labels = {}
    features_names_and_labels['features'] = features
    features_names_and_labels['labels'] = train_set_y
    features_names_and_labels['names'] = names

    pickle.dump(features, open('train_features.p', 'wb'))
    pickle.dump(features_names_and_labels, open('features_names_and_labels.p', 'wb'))

def test_features():

    f = gzip.open('../../../datasets/Mnist/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    X_test, test_set_y = test_set

    convnet = pickle.load(open('training_outputs/best_model.p', 'rb'))

    print(X_test.shape)
    test_set_x = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))
    test_features = convnet.features(test_set_x)
    print(test_features.shape)

    nb_labels = defaultdict(int)
    names = []
    for feats, label in zip(test_features, test_set_y):

        nb_labels[label] += 1
        name = '../../../datasets/Mnist/test_features/'+str(label)+'_'+str(nb_labels[label])+'.p'
        # name = str(label)+'_'+str(nb_labels[label])+'.jpg'
        print(name)
        pickle.dump(feats, open(name, 'wb'))


dataset = sys.argv[1]
train_convnet(dataset)
# train_features()
# test_features()
