import json
import keras
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(0)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
session = keras.backend.get_session()

from data import MNIST
from policy import json2policy
from models import Autoencoder, PSNR

policy_filename = "../models/policies/model_0_policy.json"
with open('../models/policies/model_0_policy.json', "r") as f:
    policy_json = json.load(f)
policy = json2policy(policy_json)

n_epochs        = 500
batch_size      = 32
n_train_samples = 50000
lr              = 0.02 
decay           = 0.1

mnist_gauss_05 = MNIST(batch_size = 32, n_train_samples = 50, noise_type = "Gaussian", intensity = 0.5)

ae_pol0 = Autoencoder(mnist_gauss_05, n_epochs = n_epochs, n_stages = 20, lr = lr, decay = decay)
ae_npol = Autoencoder(mnist_gauss_05, n_epochs = n_epochs, n_stages = 20, lr = lr, decay = decay)
ae_pol0.train(do_valid = False, policy = policy)
ae_npol.train(do_valid = False, policy = policy)

print("Performance with policy:")
print(ae_pol0.test())
print("Performance without policy:")
print(ae_npol.test())

Xbatch, Ybatch = mnist_gauss_05.get_batch(data = "Valid")

X_pol0_rec = ae_pol0.model.predict(Xbatch)
X_npol_rec = ae_npol.model.predict(Xbatch)

gauss_PSNR    = [np.round(session.run(PSNR(Ybatch[i], Xbatch[i])), 3) for i in range(3)]
rec_pol0_PSNR = [np.round(session.run(PSNR(Ybatch[i], X_pol0_rec[i])), 3) for i in range(3)]
rec_npol_PSNR = [np.round(session.run(PSNR(Ybatch[i], X_npol_rec[i])), 3) for i in range(3)]

fig, axis = plt.subplots(3, 4, figsize = (12, 4))
fontsize = 10

for i in range(3):
    axis[i, 0].imshow(Ybatch[i].reshape(28, 28), cmap = 'gray')
    axis[i, 0].axis('off')
    
    axis[i, 1].imshow(Xbatch[i].reshape(28, 28), cmap = 'gray')
    axis[i, 1].text(15, 32, "PSNR: {0:.2f}".format(gauss_PSNR[i]), horizontalalignment='center', fontsize = fontsize)
    axis[i, 1].axis('off')
    
    axis[i, 2].imshow(X_pol0_rec[i].reshape(28, 28), cmap = 'gray')
    axis[i, 2].text(15, 32, "PSNR: {0:.2f}".format(rec_pol0_PSNR[i]), horizontalalignment='center', fontsize = fontsize)
    axis[i, 2].axis('off')
    
    axis[i, 3].imshow(X_npol_rec[i].reshape(28, 28), cmap = 'gray')
    axis[i, 3].text(15, 32, "PSNR: {0:.2f}".format(rec_npol_PSNR[i]), horizontalalignment='center', fontsize = fontsize)
    axis[i, 3].axis('off')

axis[0, 0].text(-20, 14, "Image 1", horizontalalignment='center', fontsize = 12)
axis[1, 0].text(-20, 14, "Image 2", horizontalalignment='center', fontsize = 12)
axis[2, 0].text(-20, 14, "Image 3", horizontalalignment='center', fontsize = 12)

axis[0, 0].set_title('Original')
axis[0, 1].set_title('Noised')
axis[0, 2].set_title('Reconstruction w/ policy')
axis[0, 3].set_title('Reconstruction')

plt.savefig('../logs/PaperFigures/Denoising_Gauss_std05.eps')
plt.show()