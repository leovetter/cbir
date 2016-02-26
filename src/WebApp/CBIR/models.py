from django.db import models
from django.conf import settings
from skimage.io import imread
from skimage.io import imsave
import pickle
import cv2
import numpy as np
from CBIR import settings
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from skimage.color import rgb2gray

class Image(models.Model):

    img = models.ImageField(upload_to=settings.UPLOAD_PATH, default='')
    # features = PickledObjectField()
    name = models.CharField(max_length=100, default='')

    def features(self, algo, nb_bins=None):

        if algo == 'bag':

            img = imread(self.img.path)

            features = self.bag_of_words(img, nb_bins)
        elif algo == 'convnet':

            name = self.img.path.split('/')[-1].split('.')[0]
            features = pickle.load(open('../../datasets/Mnist/features/convnet/test_features/'+name+'.p', 'rb'))

            # imgs_query = []
            # for i in range(50):
            #     img = imread(self.img.path)
            #
            #     img = rgb2gray(img)
            #     img = resize(img, (28,28))
            #     imgs_query.append(img)
            # imgs_query = np.array(imgs_query)
            #
            # features = self.convnet(imgs_query)

        return features

    def bag_of_words(self, img, nb_bins):

        # img = rgb2gray(img)
        # print(img.shape)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # print(np.unique(gray))
        # print(gray.shape)
        gray = resize(gray, (28,28)) * 255

        print(np.unique(gray))

        gray = gray.astype('uint8')
        # print(img.shape)

        print('unique')
        print(np.unique(gray))

        sift = cv2.SIFT(contrastThreshold=0.01, edgeThreshold=50, sigma=0.8)
        kp, des = sift.detectAndCompute(gray,None)

        print('descriptors')
        print(des)

        model = pickle.load(open('../features/bag/output/mnist/64-cluster/train_kmeans_model.npy', 'rb'))

        features = model.predict(des)
        histo = np.bincount(features, minlength=nb_bins)

        return np.array(histo)

    def convnet(self, imgs):

        convnet = pickle.load(open('../algos/convnet/training_outputs/best_model.p', 'rb'))

        imgs = np.reshape(imgs, (imgs.shape[0], 1, imgs.shape[1], imgs.shape[2]))
        features = convnet.features(imgs)
        features = features[0]

        return features


# class Dataset(models.Model):
#
#     name = models.CharField(max_length=30)
