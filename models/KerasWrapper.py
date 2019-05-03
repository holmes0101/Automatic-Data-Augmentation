###########################################################################
# InnovR Source Code: Data Augmentation for Image Denoising              #
# Author: Eduardo Fernandes Montesuma                                    # 
# Date: 14.03.2019                                                       #
#                                                                        #
# This code proposes wrapper functions for multiple keras functionalities#
# such as layer functions and callbacks                                  #
##########################################################################

import io
import os
import json
import keras
import numpy as np
import tensorflow as tf

from PIL import Image

LOG_DIR = "./logs/autoencoder_runs"

def KerasConvlayer(x, n_filters, k_size, strides = (1, 1), batch_norm = True, name = "ConvLayer"):
    """
    Wraps the three steps needed to compute the Convlayer
    with BatchNorm and LearkyRelu activation
    """
    with tf.name_scope(name):
        pre_activations = keras.layers.Conv2D(filters = n_filters, 
                                              kernel_size = k_size, 
                                              strides = strides, 
                                              padding = 'same', activation = "linear")(x)
        if batch_norm:
            pre_activations = keras.layers.BatchNormalization(scale = False)(pre_activations)

        return keras.layers.LeakyReLU()(pre_activations)

def KerasTranspConvlayer(x, n_filters, k_size, strides = (1, 1), batch_norm = True, name = "T_ConvLayer"):
    """
    Wraps the three steps needed to compute the TranspConvlayer
    with BatchNorm and LearkyRelu activation
    """
    with tf.name_scope(name):
        pre_activations = keras.layers.Conv2DTranspose(filters = n_filters, 
                                                       kernel_size = k_size, 
                                                       strides = strides, 
                                                       padding = 'same', activation = "linear")(x)
        if batch_norm:
            pre_activations = keras.layers.BatchNormalization(scale = False)(pre_activations)

        return keras.layers.LeakyReLU()(pre_activations)

def KerasPooling(x, pool_size = (2, 2), strides = None, name = "PoolingLayer"):
    with tf.name_scope(name):
        return keras.layers.MaxPooling2D(pool_size = pool_size, strides = strides, padding = "same")(x)

def KerasSSIM(y_true, y_pred):
    """
    Wraps tensorflow SSIM loss between y_true <- ground truth
    and y_pred <- reconstructed output from input.
    """
    
    return tf.reduce_mean(1.0 - tf.image.ssim(tf.expand_dims(y_true[:, :, :, 0], -1), 
                                tf.expand_dims(y_pred[:, :, :, 0], -1), 1.0))

def KerasRMSE(y_true, y_pred):
    """
    Wraps tensorflow RMSE between y_true <- ground truth
    and y_pred <- reconstructed output from input.
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def KerasPSNR(y_true, y_pred):
    """
    Wraps tensorflow PSNR between
        y_trye <- ground truth
        y_pred <- reconstructed output
    """
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val = 1.0))


def KerasSE(y_true, y_pred):
    """
    Wraps tensorflow SE between
        y_true <- ground truth
        y_pred <- reconstructed output
    """
    return tf.reduce_sum(tf.square(y_true - y_pred))

def KerasCustomLoss(y_true, y_pred, w_ssim, w_rmse, w_psnr):
    """
    Custom Loss used in DAE
    """    
    SSIM = KerasSSIM(y_true, y_pred)
    RMSE = KerasRMSE(y_true, y_pred)
    PSNR = 1 / (1 + KerasPSNR(y_true, y_pred))
    
    return w_ssim * SSIM + w_rmse * RMSE + w_psnr * PSNR

def make_image(image_tensor):
    """
    Convert image to tf summary image
    
    https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1
    """
    
    h, w, c = image_tensor.shape
    image = Image.fromarray(image_tensor.reshape(h, w))
    output = io.BytesIO()
    image.save(output, format = 'PNG')
    image_string = output.getvalue()
    output.close()
    
    return tf.Summary.Image(height = h, width = w, colorspace = c,
                            encoded_image_string = image_string)
    
class TensorBoardImage(keras.callbacks.Callback):
    """
    Custom Keras callback concerning image writting on tensorboard.
    * OBS: call this callback BEFORE tensorboard callback.
    """
    def __init__(self, AE_object):
        super().__init__()
        # Creates auxiliary folders
        # Folder string for tensorboard files
        """
        self.folder_string = "lr({})_dec({})_a({})_b({})".format(np.round(AE_object.lr, 2),     # Learning Rate
                                                                 np.round(AE_object.decay, 2),  # Learning Rate Decay
                                                                 np.round(AE_object.w_ssim, 2), # Weight SSIM
                                                                 np.round(AE_object.w_rmse, 2)) # Weight RMSE
        """
        #self.folder_string = "lr({})_dec({})".format(np.round(AE_object.lr, 2), np.round(AE_object.decay, 2))
        self.folder_string = AE_object.model_name
        # Checks if folder already exists
        if not os.path.isdir(os.path.join(LOG_DIR, self.folder_string)):
            os.mkdir(os.path.join(LOG_DIR, self.folder_string))
        # Tensorboard objects
        self.writer = tf.summary.FileWriter(os.path.join(LOG_DIR, self.folder_string)) # Tensorboard Writer
        self.writer.add_graph(keras.backend.get_session().graph)                       # Adds model graph to tensorboard
        # Auxiliary objects
        self.AE_object = AE_object # Auto encoder model
        self.seen = 0              # Number of seen epochs
        
    def on_epoch_end(self, epoch, logs = {}):
        self.seen += 1 # Updates intern epoch counter

        # Tensorboard summaries:

        # SSIM summary
        summary_SSIM = tf.Summary(value = [tf.Summary.Value(tag = "train SSIM", simple_value = 1 - logs['KerasSSIM'])])
        self.writer.add_summary(summary_SSIM, epoch)
        # RMSE summary
        summary_RMSE = tf.Summary(value = [tf.Summary.Value(tag = "Train RMSE", simple_value = logs['KerasRMSE'])])
        self.writer.add_summary(summary_RMSE, epoch)
        # PSNR summary
        summary_PSNR = tf.Summary(value = [tf.Summary.Value(tag = "train PSNR", simple_value = logs['KerasPSNR'])])
        self.writer.add_summary(summary_PSNR, epoch)
        # Loss summary
        summary_loss = tf.Summary(value = [tf.Summary.Value(tag = "Training Loss", simple_value = logs['loss'])])
        self.writer.add_summary(summary_loss, epoch)
        # Validation Loss summary
        summary_val_loss = tf.Summary(value = [tf.Summary.Value(tag = "Validation Loss", simple_value = logs['val_loss'])])
        self.writer.add_summary(summary_val_loss, epoch)
        # Validation PSNR
        summary_val_PSNR = tf.Summary(value = [tf.Summary.Value(tag = "Validation PSNR", simple_value = logs['val_KerasPSNR'])])
        self.writer.add_summary(summary_val_PSNR, epoch)
        # Validation RMSE
        summary_val_RMSE = tf.Summary(value = [tf.Summary.Value(tag = "Validation RMSE", simple_value = logs['val_KerasRMSE'])])
        self.writer.add_summary(summary_val_RMSE, epoch)
        # Validation SSIM
        summary_val_SSIM = tf.Summary(value = [tf.Summary.Value(tag = "Validation SSIM", simple_value = 1 - logs['val_KerasSSIM'])])
        self.writer.add_summary(summary_val_SSIM, epoch)

        if self.seen % 10 == 0:
            # For each 10 epochs, we write images into tensorboard
            x, y, xd = self.AE_object.inference()

            # Converts 8 first samples to uint8
            noise    = np.uint8(255 *  x[0:8]) 
            original = np.uint8(255 *  y[0:8])
            denoised = np.uint8(255 * xd[0:8])

            # Stacking images per batch
            a = np.vstack([noise[i]    for i in range(len(noise))])
            b = np.vstack([original[i] for i in range(len(original))])
            c = np.vstack([denoised[i] for i in range(len(denoised))])

            # Stacking batches side-by-side
            img = np.hstack([a, b, c])
            img = make_image(img)
            
            summary_original = tf.Summary(value = [tf.Summary.Value(tag = "Network_Input", image = img)])            
            self.writer.add_summary(summary_original, epoch)


        return

def CustomCheckpoint(AE_object, monitor = "val_loss", save_best_only=False, 
                     save_weights_only=False, mode='auto', period=1):
    """
    Custom Checkpoint callback
    """
    """
    folder_string = "lr({})_dec({})_a({})_b({})".format(np.round(AE_object.lr, 2),     # Learning Rate
                                                        np.round(AE_object.decay, 2),  # Learning Rate Decay
                                                        np.round(AE_object.w_ssim, 2), # Weight SSIM
                                                        np.round(AE_object.w_rmse, 2)) # Weight RMSE
    """
    #folder_string = "lr({})_dec({})".format(np.round(AE_object.lr, 2), np.round(AE_object.decay, 2))
    folder_string = AE_object.model_name
    if not os.path.isdir(os.path.join(LOG_DIR, folder_string, "model_files")):
        os.mkdir(os.path.join(LOG_DIR, folder_string, "model_files"))
        # Checks existence of model .json file
        exists = os.path.isfile(os.path.join(LOG_DIR, folder_string, "model_files", "model.json"))
        if not exists:
            model_json = AE_object.model.to_json() # Converts model to JSON
            with open(os.path.join(LOG_DIR, folder_string, "model_files", "model.json"), "w") as json_file:
                json_file.write(model_json) # Write to file
    
    dir_path = os.path.join(LOG_DIR, folder_string, "model_files")

    return keras.callbacks.ModelCheckpoint(os.path.join(dir_path, "weights.hdf5"),
                                           monitor, save_best_only, save_weights_only, mode, period)

def load_previous_run(path = "/home/efernand/training_output/KerasAutoEncoder_runnings"):
    """
    Load keras pretained model
    """

    model_path   = os.path.join(path, "model.json")
    weights_path = os.path.join(path, "weights.h5")

    with open(model_path, "r") as json_file:
        arch  = json.load(json_file)
        model = keras.models.model_from_json(json.dumps(arch))

    model.load_weights(weights_path) 

    return model
