import numpy as np
import random
from skimage.io import imread
from skimage.transform import resize
exec(open('../pascal/constants.py').read())
DATA_DIVIDE = 3

def load_images(resize_img, filename):

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

def load_pascal_set(resize, *labels):

    all_train_x = None
    all_train_y = None
    all_train_images = None
    all_val_x = None
    all_val_y = None
    all_val_images = None
    for label in labels:

        train_x, train_images, val_x, val_images = load_images(resize, label)

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
