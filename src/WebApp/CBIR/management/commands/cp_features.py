from django.core.management.base import BaseCommand, CommandError
from algos import *
import pickle
import sys
import numpy as np

sys.path.append('../convnet/')
# from PascalNet import PascalNet
from skimage.viewer import ImageViewer
from skimage.io import imread
from skimage.transform import resize

from CBIR.models import Image

class Command(BaseCommand):

    def add_arguments(self, parser):

        parser.add_argument('--algo')

    def handle(self, *args, **options):

        if options['algo'] == 'bag':

            bag_of_words()


    def two_labels_data(self):

        print('Loading datasets...')

        test_files = np.loadtxt('/Users/leo/Documents/Projets/image_retrieval/datasets/Pascal/'+'VOC2012_test/ImageSets/Main/test.txt', dtype=bytes).astype(str)
        test_files = test_files[0:100]

        test_set_x = None
        for test_file in test_files:

            img = imread('/Users/leo/Documents/Projets/image_retrieval/datasets/Pascal/'+'VOC2012_test/JPEGImages/'+test_file+'.jpg')

            img = resize(img, (200,285,3))

            if test_set_x is None:
                test_set_x = np.concatenate(([img],))
            else:
                test_set_x = np.concatenate((test_set_x, [img]))


        test_set_x = np.array(test_set_x)
        print('size test set : %d' % len(test_set_x))

        test_set_x = np.transpose(test_set_x, (0,3,1,2))

        return test_set_x, test_files

    def convnet():

        convnet = pickle.load(open('../pascal/best_model.p', 'rb'))
        images, names = self.two_labels_data()

        features = convnet.get_features(images)

        for feature, name in zip(features, names):

            img = Image(
                        features=feature,
                        name = name
                        )
            img.save()
