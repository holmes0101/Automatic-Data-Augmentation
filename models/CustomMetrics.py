import tensorflow as tf


def SSIM(y_true, y_pred):
    """
    Wraps tensorflow SSIM loss between y_true <- ground truth
    and y_pred <- reconstructed output from input.
    """
    
    return tf.reduce_mean(1.0 - tf.image.ssim(tf.expand_dims(y_true[:, :, :, 0], -1), 
                                tf.expand_dims(y_pred[:, :, :, 0], -1), 1.0))

def RMSE(y_true, y_pred):
    """
    Wraps tensorflow RMSE between y_true <- ground truth
    and y_pred <- reconstructed output from input.
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def PSNR(y_true, y_pred):
    """
    Wraps tensorflow PSNR between
        y_trye <- ground truth
        y_pred <- reconstructed output
    """
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val = 1.0))

def CustomLoss(y_true, y_pred, w_ssim, w_rmse, w_psnr):
    """
    Custom Loss used in DAE
    """    
    ssim = SSIM(y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    psnr = 1 / (1 + PSNR(y_true, y_pred))
    
    return w_ssim * ssim + w_rmse * rmse + w_psnr * psnr