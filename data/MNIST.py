import numpy as np
from mnist import MNIST


def get_mnist_data(valid=None):
    mdata = MNIST('data', return_type='numpy')

    train_data, train_labels = mdata.load_training()
    test_data, test_labels = mdata.load_testing()
    train_data = train_data.reshape((60000, 1, 28, 28))
    test_data = test_data.reshape((10000, 1, 28, 28))

    if valid is not None:
        n_valid = int(valid * 60000)
        indexes = np.array(range(60000))
        np.random.shuffle(indexes)

        valid_X = train_data[indexes[0:n_valid]]
        valid_y = train_labels[indexes[0:n_valid]]
        train_X = train_data[indexes[n_valid:]]
        train_y = train_labels[indexes[n_valid:]]
        return train_X, train_y, valid_X, valid_y, test_data, test_labels

    return train_data, train_labels, test_data, test_labels
