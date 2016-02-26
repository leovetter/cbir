from deepnet import MLP
import gzip
import pickle
f = gzip.open('../../datasets/Mnist/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding='bytes')
f.close()
X_train, y_train = train_set
X_valid, y_valid = valid_set
mlp = MLP(X_train, y_train, X_valid, y_valid, 200)
mlp.fit(5, 0.1, 0.9, 0, 0, True)
