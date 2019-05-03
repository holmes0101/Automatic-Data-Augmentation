import io
import os
import keras
import numpy as np
import tensorflow as tf
import PIL.Image as Image

LOG_DIR = "../logs/autoencoder_runs"

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
        # Folder string for tensorboard files
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