import os
import sys
import json
import keras
import numpy as np

sys.path.append(os.path.abspath('../'))

from models import Autoencoder
from data import MNIST as images_dataset
#from data import EMNIST as images_dataset
from transformations import json2policy

with open('./policies/model_0_policy.json', "r") as f:
    policy_json = json.load(f)
policy = json2policy(policy_json)

# Autoencoder definition
n_epochs  = 500
lr        = 0.02
decay     = 0.1 
dataset_type   = "MNIST"
noise     = "Gaussian"
std_dev   = 0.5
n_samples_vec = np.round(np.linspace(32, 16000, 100)).astype(int)


for n_samples in n_samples_vec:
    print("Running with {} samples".format(n_samples))
    session = keras.backend.get_session()
    dataset = images_dataset(batch_size = 32, n_train_samples = n_samples, noise_type = noise, intensity = std_dev)
    
    ae_pol0 = Autoencoder(dataset, n_epochs = n_epochs, n_stages = 20, test_runs = 1, lr = lr, decay = decay, model_name = "size:{}_pol0".format(n_samples))
    ae_npol = Autoencoder(dataset, n_epochs = n_epochs, n_stages = 20, test_runs = 1, lr = lr, decay = decay, model_name = "size:{}_npol".format(n_samples))
    
    ae_pol0.train(do_valid = False, policy = policy)
    ae_npol.train(do_valid = False)
    
    
    def evaluate_model(dataset, model_pol, model_no_pol, batch_size = 32):
        score_ssim1 = []
        score_rmse1 = []
        score_psnr1 = []
    
        score_ssim2 = []
        score_rmse2 = []
        score_psnr2 = []
    
        Xtest = dataset.x_test
        Ytest = dataset.y_test
    
        for i in range( len(Xtest) // batch_size ):
            Xbatch = Xtest[i * batch_size : (i + 1) * batch_size]
            Ybatch = Ytest[i * batch_size : (i + 1) * batch_size]
        
            [loss1, ssim1, rmse1, psnr1] = ae_pol0.model.evaluate(x = Xbatch, y = Ybatch, batch_size = None, verbose = 0)
            [loss2, ssim2, rmse2, psnr2] = ae_npol.model.evaluate(x = Xbatch, y = Ybatch, batch_size = None, verbose = 0)
    
            print("Batch {}/{}".format(i, len(Xtest) // batch_size))
    
            # Scores with policy
            score_ssim1.append(ssim1)
            score_rmse1.append(rmse1)
            score_psnr1.append(psnr1)
            # Scores without policy
            score_ssim2.append(ssim2)
            score_rmse2.append(rmse2)
            score_psnr2.append(psnr2)
    
        return [np.mean(score_ssim1), np.std(score_ssim1),
                np.mean(score_rmse1), np.std(score_rmse1),
                np.mean(score_psnr1), np.std(score_psnr1),
                np.mean(score_ssim2), np.std(score_ssim2),
                np.mean(score_rmse2), np.std(score_rmse2),
                np.mean(score_psnr2), np.std(score_psnr2)]
                
    [avg_ssim1, std_ssim1, 
     avg_rmse1, std_rmse1,
     avg_psnr1, std_psnr1,
     avg_ssim2, std_ssim2,
     avg_rmse2, std_rmse2,
     avg_psnr2, std_psnr2] = evaluate_model(dataset, ae_pol0.model, ae_npol.model)
    
    
    with open("../logs/PerformanceDataCurve/test_{}_{}_stddev{}.txt".format(dataset_type, noise, std_dev), "a+") as f:
        print("Writting on file")
        f.write("{}\t, {}\t, {}\t, {}\t, {}\t, {}\t, {}\t, {}\n".format(n_samples, 1, avg_ssim1, std_ssim1, avg_rmse1, std_rmse1, avg_psnr1, std_psnr1))
        f.write("{}\t, {}\t, {}\t, {}\t, {}\t, {}\t, {}\t, {}\n".format(n_samples, 0, avg_ssim2, std_ssim2, avg_rmse2, std_rmse2, avg_psnr2, std_psnr2))
        f.write("\n")
