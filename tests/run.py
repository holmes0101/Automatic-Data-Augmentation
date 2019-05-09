import sys

sys.path.append('../')

# Python modules
import tensorflow as tf
import keras.backend as backend

# My modules
from data import MNIST
from controller import Controller, ModelSaver


tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
backend.set_session(session)

CONTROLLER_EPOCHS = int(1e+6)   # Number of training iterations for controller
N_TOKEEP          = 5           # Number of models to keep

# Modules
dataset     = MNIST(batch_size = 128)                                				# Data loader module
pol_saver   = ModelSaver(N_TOKEEP)                                                  # Policy saver module
controller  = Controller(session, dataset, pol_saver, n_epochs = CONTROLLER_EPOCHS) # AutoAugment Controller module
controller.train()

