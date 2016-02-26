import numpy as np
import random
from skimage.io import imread
from skimage.transform import resize
DATA_DIVIDE = 3
BASE_DIR_PASCAL='../../../datasets/Pascal/'
import sys
import glob
from skimage.io import imread
from skimage.transform import resize

def load_pascal_images(set_name, resize_img, filename):

    if set_name=="train":
        train_files = np.loadtxt(BASE_DIR_PASCAL+'VOC2012/ImageSets/Main/'+filename+'_train.txt', dtype=bytes).astype(str)
        train_files = [name[0] for name in train_files if int(name[1]) == 1]
        random.shuffle(train_files)
        train_files = train_files[0:int(len(train_files)/DATA_DIVIDE)]

        val_files = np.loadtxt(BASE_DIR_PASCAL+'VOC2012/ImageSets/Main/'+filename+'_val.txt', dtype=bytes).astype(str)
        val_files = [name[0] for name in val_files if int(name[1]) == 1]
        random.shuffle(val_files)
        val_files = val_files[0:int(len(val_files)/DATA_DIVIDE)]

        train_length = len(train_files)
        train_set_x = []
        train_set_y = []
        for train_file in train_files:

            img = imread(BASE_DIR_PASCAL+'VOC2012/JPEGImages/'+train_file+'.jpg')

            if resize_img:
                img = resize(img, (200,285,3))

            train_set_x.append(img)
            train_set_y.append(train_file)

        val_length = len(val_files)
        val_set_x = []
        val_set_y = []
        for val_file in val_files:

            img = imread(BASE_DIR_PASCAL+'VOC2012/JPEGImages/'+val_file+'.jpg')

            if resize_img:
                img = resize(img, (200,285,3))

            val_set_x.append(img)
            val_set_y.append(val_file)

        return train_set_x, train_set_y, val_set_x, val_set_y

    elif set_name=="test":
        test_files = np.loadtxt(BASE_DIR_PASCAL+'VOC2012_TEST/ImageSets/Main/'+filename+'_test.txt', dtype=bytes).astype(str)
        test_files = [name[0] for name in test_files if int(name[1]) == 1]
        random.shuffle(test_files)
        test_files = test_files[0:int(len(test_files)/DATA_DIVIDE)]

        test_length = len(test_files)
        test_set_x = []
        test_set_y = []
        for test_file in test_files:

            img = imread(BASE_DIR_PASCAL+'VOC2012_TEST/JPEGImages/'+test_file+'.jpg')

            if resize_img:
                img = resize(img, (200,285,3))

            test_set_x.append(img)
            test_set_y.append(test_file)

        return test_set_x, test_set_y

def load_pascal_set(set_name,resize, *labels):

    if set_name=="train":
        all_train_x = None
        all_train_y = None
        all_train_images = None
        all_val_x = None
        all_val_y = None
        all_val_images = None
        for label in labels:

            train_x, train_images, val_x, val_images = load_pascal_images(set_name, resize, label)

            train_y = np.array(train_images)
            train_y.fill(label)
            val_y = np.array(val_images)
            val_y.fill(label)

            if all_train_x is None:
                all_train_x = train_x
                all_train_y = train_y
                all_train_images = train_images
                all_val_x = val_x
                all_val_y = val_y
                all_val_images = val_images
            else:
                all_train_x = np.concatenate([all_train_x, train_x])
                all_train_y = np.concatenate([all_train_y, train_y])
                all_train_images = np.concatenate([all_train_images, train_images])
                all_val_x = np.concatenate([all_val_x, val_x])
                all_val_y = np.concatenate([all_val_y, val_y])
                all_val_images = np.concatenate([all_val_images, val_images])

        return [
                (all_train_x, all_train_y, all_train_images),
                (all_val_x, all_val_y, all_val_images)
                ]

    elif set_name=="test":

        all_test_x = None
        all_test_y = None
        all_test_images = None
        for label in labels:

            test_x, test_images = load_pascal_images(set_name, resize, label)
            test_y = np.array(test_images)
            print(label)
            print(test_images)
            print(test_x)
            print(test_y)
            test_y.fill(label)

            if all_test_x is None:
                all_test_x = test_x
                all_test_y = test_y
                all_test_images = test_images
            else:
                all_test_x = np.concatenate([all_test_x, test_x])
                all_test_y = np.concatenate([all_test_y, test_y])
                all_test_images = np.concatenate([all_test_images, test_images])

        return [
                (all_test_x, all_test_y, all_test_images),
                ]

def load_paris_set():

    all_train_images = glob.glob('../../../datasets/Paris/train/*')
    random.shuffle(all_train_images)
    all_train_images = all_train_images[0:int(len(all_train_images)*0.3)]
    all_train_x = []
    all_train_y = []
    for train_image in all_train_images:

        img = imread(train_image)
        img = resize(img, (100, 150, 3))
        img = np.transpose(img, (2,0,1))
        all_train_x.append(img)

        label = train_image.split('/')[-1].split('_')[0]
        all_train_y.append(label)

    all_train_x = np.array(all_train_x)
    all_train_y = np.unique(all_train_y, return_inverse=True)[1]

    all_val_images = glob.glob('../../../datasets/Paris/valid/*')
    random.shuffle(all_val_images)
    #all_val_images = all_val_images[0:len(all_val_images)*0.8]
    all_val_x = []
    all_val_y = []
    for val_image in all_val_images:

        img = imread(val_image)
        img = resize(img, (100, 150, 3))
        img = np.transpose(img, (2,0,1))
        all_val_x.append(img)

        label = val_image.split('/')[-1].split('_')[0]
        all_val_y.append(label)

    all_val_x = np.array(all_val_x)
    all_val_y = np.unique(all_val_y, return_inverse=True)[1]

    return [
            (all_train_x, all_train_y, all_train_images),
            (all_val_x, all_val_y, all_val_images)
            ]
