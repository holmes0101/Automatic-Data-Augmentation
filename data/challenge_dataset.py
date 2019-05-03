import os
import sys

sys.path.append(os.path.abspath('../'))

import skimage.io
import numpy as np
import pandas as pd
import tensorflow as tf

from functools import partial
from data import AbstractDataset

def filter_filenames(filenames, filter_set):
    for filename in filenames:
        sub_filename = str.split(os.path.splitext(filename)[0], '_')
        key = sub_filename[0]+ '_' + sub_filename[1]
        if key in filter_set:
            yield filename

def _gen_np_arrays(path, filenames, isref = False):
    if isref:
        return np.expand_dims(np.stack(
                skimage.img_as_ubyte(skimage.io.imread(os.path.join(path, "ref/", filename), as_grey=True))
                for filename in filenames), axis=3)
    else:
        return np.expand_dims(np.stack(
                skimage.img_as_ubyte(skimage.io.imread(os.path.join(path, "in_mul/", filename), as_grey=True))
                for filename in filenames), axis=3)

class Challenge(AbstractDataset):
    def __init__(self, tf_sess, config_json, n_train_samples = -1):
        self.tf_sess = tf_sess
        self.batch_size = config_json["batch_size"]
        self.config_json = config_json
        self.n_train_samples = n_train_samples
        self.build_dataset()

        super().__init__()

    def build_dataset(self):
        """
        This function build the dataset based on tensorflow dataset API
        """           
        # Filtering files
        filter = self.config_json["filter"]
        if filter:
            csv_path = os.path.join(self.config_json["root_path"], "content.csv")
            csv_data = pd.read_csv(csv_path)
            sub_datas = csv_data[csv_data.solid_bg == True]
            filter_set = set(sub_datas.apply(lambda row: str(row['timestamp']) + '_' + str(row['sample_index']), axis=1))
            # Filenames
            filenames = {
                'train' : list(filter_filenames(os.listdir(self.config_json["train_path"] + "in/"), filter_set)),
                'valid' : list(filter_filenames(os.listdir(self.config_json["valid_path"] + "in/"), filter_set)),
                'test'  : list(filter_filenames(os.listdir(self.config_json["test_path"] + "in/"), filter_set))
            }
        else:
            print("Here")
            filenames = {
                "train" : os.listdir(self.config_json["train_path"] + "in/")[0 : self.n_train_samples],
                "valid" : os.listdir(self.config_json["valid_path"] + "in/"),
                "test"  : os.listdir(self.config_json["test_path"]  + "in")
            }
        # Saves image shape
        self.image_shape = np.shape(
            np.expand_dims(skimage.io.imread(os.path.join(self.config_json["train_path"] + "/in/", filenames['train'][0])), axis = -1)
        )

        self.num_samples = {
            "train" : len(filenames["train"]),
            "valid" : len(filenames["valid"]),
            "test"  : len(filenames["test"])
        }

        print(self.num_samples)
        # Mapping from filenames to tf tensors
        def _tf_map(filename, path):
            _in_file  = tf.io.read_file(path + "in/" + filename)
            _in_image = tf.image.decode_image(_in_file, 1)
            _in_image = tf.image.resize_image_with_crop_or_pad(_in_image, self.image_shape[0], self.image_shape[1])
            _in_image.set_shape(self.image_shape)

            _out_file  = tf.io.read_file(path + "ref/" + filename)
            _out_image = tf.image.decode_image(_out_file, 1)
            _out_image = tf.image.resize_image_with_crop_or_pad(_out_image, self.image_shape[0], self.image_shape[1])
            _out_image.set_shape(self.image_shape)

            return (_in_image, _out_image)
        map_train = partial(_tf_map, path = self.config_json["train_path"]) # Train mapping function
        map_valid = partial(_tf_map, path = self.config_json["valid_path"]) # Valid mapping function
        map_test  = partial(_tf_map, path = self.config_json["test_path"])  # Test mapping function
        # Building dataset
        with tf.name_scope("Dataset"):
            train_dataset = tf.data.Dataset.from_tensor_slices(filenames['train']).shuffle(buffer_size=128).repeat().map(
                map_train, num_parallel_calls=16).batch(self.batch_size).prefetch(4)

            valid_dataset = tf.data.Dataset.from_tensor_slices(filenames['valid']).shuffle(buffer_size=128).repeat().map(
                map_valid, num_parallel_calls=16).batch(self.batch_size).prefetch(4)

            test_dataset = tf.data.Dataset.from_tensor_slices(filenames['test']).shuffle(buffer_size=128).repeat().map(
                map_test, num_parallel_calls=16).batch(self.batch_size).prefetch(4)

            self.datasets = {
                'train' : train_dataset,
                'valid' : valid_dataset,
                'test'  : test_dataset
            }

            self.handle = tf.placeholder(tf.string, shape = [])           # String handler placeholder
            self.iterator = tf.data.Iterator.from_string_handle(self.handle, 
                                                                output_types = self.datasets["train"].output_types, 
                                                                output_shapes = self.datasets["train"].output_shapes)
            self.x, self.y = self.iterator.get_next()                     # Next train batch element
            self.x = tf.cast(self.x, tf.float32) * (1.0 / 255.0)          # Casting inputs to float
            self.y = tf.cast(self.y, tf.float32) * (1.0 / 255.0)          # Casting references to float

            with tf.name_scope("Train_Iterator"):
                self.train_iterator = train_dataset.make_one_shot_iterator()
                self.train_iterator_handle = self.tf_sess.run(self.train_iterator.string_handle())
            with tf.name_scope("Test_Iterator"):
                self.test_iterator  = test_dataset.make_one_shot_iterator()
                self.test_iterator_handle = self.tf_sess.run(self.test_iterator.string_handle())
            with tf.name_scope("Valid_Iterator"):
                self.valid_iterator  = valid_dataset.make_one_shot_iterator()
                self.valid_iterator_handle = self.tf_sess.run(self.valid_iterator.string_handle())

    def get_input_shape(self):
        """
        Returns the shape of dataset images
        """
        return self.image_shape

    def get_batch(self, data = "Train", policy = None):
        # Data handle definition
        if data == "Train": 
            batch_handle = self.train_iterator_handle
        if data == "Valid":
            batch_handle = self.valid_iterator_handle
        if data == "Test":
            batch_handle = self.test_iterator_handle
        
        # Runs tensors to retrieve numpy arrays
        _x, _y = self.tf_sess.run([self.x, self.y], feed_dict = {self.handle : batch_handle})
        # Gets shapes
        b, h, w, c = np.shape(_x)
        # If training, apply subpolicies
        if data == "Train" and policy:
            _x_transformations = [_x] # Input transformed images
            _y_transformations = [_y] # References transformed images
            # Applying subpolicies to the batch
            for i in range(b):
                xi, yi = np.expand_dims(_x[i], axis = 0), np.expand_dims(_y[i], axis = 0)
                subpolicy = np.random.choice(policy)
                xi_t, yi_t = subpolicy(xi, yi)

                _x_transformations.extend([xi, xi_t])
                _y_transformations.extend([yi, yi_t])
            _x = np.reshape(np.array(_x_transformations), [len(_x_transformations), h, w, c])
            _y = np.reshape(np.array(_y_transformations), [len(_x_transformations), h, w, c])

        return _x, _y
