from django.core.management.base import BaseCommand, CommandError
from load_data import load_pascal_set
import numpy as np
import random
from deepnet import ConvNet

class Command(BaseCommand):

    # def add_arguments(self, parser):
    #
    #     parser.add_argument('--algo')

    def handle(self, *args, **options):

            self.train_convnet()

    def train_convnet(self):

        datasets = load_pascal_set(True, 'aeroplane', 'bicycle', 'car', 'bird', 'boat', 'person')
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

        print('start training')
        print(train_set_x.shape)
        convnet = ConvNet(train_set_x, train_set_y, val_set_x, val_set_y, 50)
        convnet.fit(curves=False, max_epoch=100)
