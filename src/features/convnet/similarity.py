from skimage.io import imread
from skimage.viewer import ImageViewer
import sys
import pickle
import numpy as np
from scipy.spatial.distance import euclidean

def query(img_name):

    imgs_query = []
    for i in range(50):

        img_query = imread('../../../datasets/Mnist/images/test/'+img_name+'.jpg')
        imgs_query.append(img_query)
    imgs_query = np.array(imgs_query)

    imgs_query = np.reshape(imgs_query, (imgs_query.shape[0], 1, 28, 28))

    convnet = pickle.load(open('training_outputs/best_model.p', 'rb'))

    features_query = convnet.features(imgs_query)
    features_query = features_query[0]

    db_features = pickle.load(open('../convnet/train_features.p', 'rb'))

    id_best_features = euclidean_distances(features_query, db_features)

    features_names_and_labels = pickle.load(open('features_names_and_labels.p', 'rb'))
    names = features_names_and_labels['names']

    best_imgs = np.array(names)[id_best_features]

    print(best_imgs)

    # images = []
    # for features_and_name in best_features:
    #
    #     img_path = {
    #         'path': settings.STATIC_URL + 'images/Pascal/VOC2012/JPEGImages/'+features_and_name['name']+'.jpg'
    #     }
    #     images.append(img_path)

def euclidean_distances(features_query, db_features):

    distances = []
    for features in db_features:

        distance = euclidean(features_query, features)
        distances.append(distance)

    return  np.array(distances).argsort()[:100]

img_name = sys.argv[1]
query(img_name)
