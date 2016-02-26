from django.conf import settings
from scipy.spatial.distance import euclidean
import pickle
import numpy as np
import gzip
import sys
sys.path.append('../distance/')
from similarity import oasis_similarity, euclidean_similarity

class ImageSearcher():

    def similar_images(self, img, algo):

        if algo == 'ConvNet':

            features = img.features('convnet')
            db_features = pickle.load(open('../../datasets/Mnist/features/convnet/train_features.p', 'rb'))

            # id_best_features = euclidean_distances(features, db_features)
            id_best_features = oasis_similarity('mnist', 'convnet', features, db_features)

            features_names_and_labels = pickle.load(open('../../datasets/Mnist/features/convnet/features_names_and_labels.p', 'rb'))
            names = features_names_and_labels['names']

            best_imgs = np.array(names)[id_best_features]
            # images = self.convnet_oasis_similarity(features, db_features, names)

        elif algo == 'BOW':

            features = img.features('bag', 64)
            samples = np.array(pickle.load(open('../../datasets/Mnist/features/bag/train_features.p', 'rb')))

            db_features = []
            names = []
            for sample in samples:
                db_features.append(sample['features'])
                # names.append(sample['label'])
                names.append(sample['name'])
            db_features = np.array(db_features)

            # id_best_features = self.euclidean_distances(features, db_features)
            id_best_features = oasis_similarity('mnist', 'bag', features, db_features)

            # images = convnet_oasis_similarity(features, db_features, names)

        best_imgs = np.array(names)[id_best_features]
        images = []
        for img in best_imgs:

            print(img)
            img = img.replace('jpeg', 'jpg')
            img_path = {
                'path': settings.STATIC_URL + 'images/Mnist/' + img
                }
            images.append(img_path)

        return images
