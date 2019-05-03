import numpy as np
from imgaug import augmenters as iaa

h, w, c = 28, 28, 1

def Shear(x, y, mag):
    _z = np.concatenate([x, y], axis = 2)
    shear_op = iaa.Affine(shear = mag)
    _z = shear_op.augment_images(_z)

    return (_z[:, :, 0 : h, :], _z[:, :, h :, :])

def Rotate(x, y, mag):
    _z = np.concatenate([x, y], axis = 2)
    rotate_op = iaa.Affine(rotate = mag)
    _z = rotate_op.augment_images(_z)

    return (_z[:, :, 0 : h, :], _z[:, :, h :, :])

def ElasticDeform(x, y, mag):
    _z = np.concatenate([x, y], axis = 2)
    elastic_op = iaa.ElasticTransformation(alpha = mag, sigma = 2)
    _z = elastic_op.augment_images(_z)

    return (_z[:, :, 0 : h, :], _z[:, :, h :, :])

def Crop(x, y, mag):
    _z = np.concatenate([x, y], axis = 2)
    crop_op = iaa.Crop(percent = mag)
    _z = crop_op.augment_images(_z)

    return (_z[:, :, 0 : h, :], _z[:, :, h :, :])    

def TranslateX(x, y, mag):
    _z = np.concatenate([x, y], axis = 2)
    translate_op = iaa.Affine(translate_px = {"x" : int(mag)})
    _z = translate_op.augment_images(_z)    

    return (_z[:, :, 0 : h, :], _z[:, :, h :, :])  

def TranslateY(x, y, mag):
    _z = np.concatenate([x, y], axis = 2)
    translate_op = iaa.Affine(translate_px = {"y" : int(mag)})
    _z = translate_op.augment_images(_z)    

    return (_z[:, :, 0 : h, :], _z[:, :, h :, :])  

def FlipX(x, y, mag):
    _z = np.concatenate([x, y], axis = 2)
    flip_op = iaa.Fliplr(1.0)
    _z = flip_op.augment_images(_z)    

    return (_z[:, :, h :, :], _z[:, :, 0 : h, :])  

def FlipY(x, y, mag):
    _z = np.concatenate([x, y], axis = 2)
    flip_op = iaa.Flipud(1.0)
    _z = flip_op.augment_images(_z)    

    return (_z[:, :, 0 : h, :], _z[:, :, h :, :]) 

all_transforms = [
    {'name': 'Shear'          , 'func': Shear               , 'minval':   -20, 'maxval':  20},   # 1
    {'name': 'TranslateX'     , 'func': TranslateX          , 'minval':    -5, 'maxval':   5},   # 2
    {'name': 'TranslateY'     , 'func': TranslateY          , 'minval':    -5, 'maxval':   5},   # 3
    {'name': 'Rotate'         , 'func': Rotate              , 'minval':   -15, 'maxval':  15},   # 4
    {'name': 'FlipX'          , 'func': FlipX               , 'minval':   0.0, 'maxval':   1},   # 5
    {'name': 'FlipY'          , 'func': FlipY               , 'minval':   0.0, 'maxval':   1},   # 6
    {'name': 'Elastic Deform' , 'func': ElasticDeform       , 'minval':     0, 'maxval':   7},   # 7
]