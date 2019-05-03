import numpy as np

from data import AbstractDataset
from keras.datasets import mnist
from skimage.util import random_noise

class MNIST(AbstractDataset):
    def __init__(self, batch_size = 32, n_train_samples = 50000, noise_type = "Gaussian", intensity = 0.3):
        #np.random.seed(0)
        (x_train, labels_train), (x_test, labels_test) = mnist.load_data()

        self.batch_size = batch_size
        self.image_shape = x_train[0].shape

        x_valid = x_train[50000:]
        x_train = x_train[0 : n_train_samples]

        self.y_train = np.reshape(x_train, [n_train_samples, 28, 28, 1]).astype(np.float32) / 255.0
        self.y_valid = np.reshape(x_valid, [10000, 28, 28, 1]).astype(np.float32) / 255.0
        self.y_test  = np.reshape(x_test , [10000, 28, 28, 1]).astype(np.float32) / 255.0

        if noise_type == "Gaussian":
            self.x_train = self.y_train + (intensity ** 2) * np.random.randn(*self.y_train.shape)
            self.x_valid = self.y_valid + (intensity ** 2) * np.random.randn(*self.y_valid.shape)
            self.x_test  = self.y_test  + (intensity ** 2) * np.random.randn(*self.y_test.shape)
        if noise_type == "s&p":
            self.x_train = random_noise(self.y_train, mode = "s&p", amount = intensity)
            self.x_valid = random_noise(self.y_valid, mode = "s&p", amount = intensity)
            self.x_test  = random_noise(self.y_test, mode = "s&p", amount = intensity)

    def get_input_shape(self):
        """
        Returns the shape of dataset images
        """
        return self.image_shape

    def get_dataset_size(self):
        return {
            "Train": self.x_train.shape,
            "Valid": self.x_valid.shape,
            "Test" : self.x_test.shape,
            "BatchSize": self.batch_size
        }

    def get_batch(self, data = "Train", policy = None):
        if data == "Train":
            idx = np.arange(0, len(self.x_train))
            np.random.shuffle(idx)
            x = self.x_train[idx[0 : self.batch_size]]
            y = self.y_train[idx[0 : self.batch_size]]
            if policy:
                b, h, w, c = np.shape(x)
                _x_transformations = [] # Input transformed images
                _y_transformations = [] # References transformed images

                for i in range(b):
                    xi = np.expand_dims(x[i], axis = 0)
                    yi = np.expand_dims(y[i], axis = 0)
                    subpolicy = np.random.choice(policy)
                    xi_t, yi_t = subpolicy(xi, yi)

                    _x_transformations.extend([xi, xi_t])
                    _y_transformations.extend([yi, yi_t])
                x = np.reshape(np.array(_x_transformations), [len(_x_transformations), h, w, c])
                y = np.reshape(np.array(_y_transformations), [len(_x_transformations), h, w, c])
        elif data == "Valid":
            idx = np.arange(0, len(self.x_test))
            np.random.shuffle(idx)
            x = self.x_valid[idx[0 : self.batch_size]]
            y = self.y_valid[idx[0 : self.batch_size]]
        elif data == "Test":
            idx = np.arange(0, len(self.x_test))
            np.random.shuffle(idx)
            x = self.x_test[idx[0 : self.batch_size]]
            y = self.y_test[idx[0 : self.batch_size]]
        else:
            return None
        return x, y
    