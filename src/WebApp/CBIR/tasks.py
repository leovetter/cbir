from __future__ import absolute_import

from celery.decorators import task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

@task(name="fit")
def fit(datasets_name, batch_size, max_epoch, learning_rate, momentum_rate, weight_decay, lambda_1, curves):

    from deepnet import MLP
    # from deepnet import ConvNet
    import gzip
    import pickle
    import numpy as np

    logger.info(datasets_name)
    if datasets_name == 'Mnist':
        f = gzip.open('../../datasets/Mnist/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(f,encoding='bytes')
        f.close()

    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    X_test, y_test = test_set

    # X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
    # X_valid = np.reshape(X_valid, (X_valid.shape[0], 1, 28, 28))
    # X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

    y_test = np.array(y_test)

    mlp = MLP(X_train, y_train, X_valid, y_valid, batch_size)
    mlp.fit(max_epoch, learning_rate, momentum_rate, weight_decay, lambda_1, curves)
