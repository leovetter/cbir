from load_data import load_pascal_set
import cv2
import sys
from sklearn.cluster import KMeans
import numpy as np
import pickle
from skimage.viewer import ImageViewer
from CBIR import settings

def check(train_set_x, train_set_y, val_set_x, val_set_y):

    print(train_set_y[0])
    viewer = ImageViewer(train_set_x[0])
    viewer.show()

    print(train_set_y[10])
    viewer = ImageViewer(train_set_x[10])
    viewer.show()

    print(train_set_y[32])
    viewer = ImageViewer(train_set_x[32])
    viewer.show()

    print(train_set_y[56])
    viewer = ImageViewer(train_set_x[56])
    viewer.show()

    print(train_set_y[76])
    viewer = ImageViewer(train_set_x[76])
    viewer.show()

    print(train_set_y[119])
    viewer = ImageViewer(train_set_x[119])
    viewer.show()

    print(val_set_y[5])
    viewer = ImageViewer(val_set_x[5])
    viewer.show()

    print(val_set_y[13])
    viewer = ImageViewer(val_set_x[13])
    viewer.show()

    print(val_set_y[32])
    viewer = ImageViewer(val_set_x[32])
    viewer.show()

    print(val_set_y[51])
    viewer = ImageViewer(val_set_x[51])
    viewer.show()

    print(val_set_y[75])
    viewer = ImageViewer(val_set_x[75])
    viewer.show()

    print(val_set_y[145])
    viewer = ImageViewer(val_set_x[145])
    viewer.show()


def bag_of_words():

    datasets = load_pascal_set(False, 'aeroplane', 'bicycle', 'car', 'bird', 'boat', 'person')
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

    model = KMeans(n_clusters=settings.NB_CLUSTER)
    model.fit(all_descriptors)

    print('Compute histos')
    features = []
    for descriptors, img_name in zip(desc_by_img, train_images):

        feats = model.predict(descriptors)
        print(feats)
        histo = np.bincount(feats, minlength=settings.NB_CLUSTER)
        print(histo)
        sys.exit('')
        feat_and_name = {}
        feat_and_name['features'] = histo
        feat_and_name['name'] = img_name
        features.append(feat_and_name)

        pickle.dump(features, open('pascal_train_features.npy', 'wb'))
        pickle.dump(model, open('pascal_train_model.npy', 'wb'))
