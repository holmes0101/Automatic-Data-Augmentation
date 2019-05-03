import os
import sys
import time
import numpy as np
import tensorflow as tf

sys.path.append('../')

# Keras Modules
import keras.layers as layers
import keras.models as models
import keras.backend as backend
import keras.initializers as initializers

from keras import backend as K
from models import Autoencoder
from datetime import timedelta
from transformations.policy import N_OPS, N_SUBPOL, N_TYPES, N_PROBS, N_MAG, Operation, Subpolicy

LOG_DIR = '../logs/autoaugment_runs'
INPUT_SHAPE = (N_OPS * (N_TYPES + N_PROBS + N_MAG), 1)
N_UNITS = 100

def float2uint8(img):
    """ Converts float img to uint8 """
    return (255 * img).astype(np.uint8)

def expand_dims(layer):
    """ Custom expand_dims layer"""
    return backend.expand_dims(layer, axis = -1)

def softmax_summaries():
    """
        This function constructs the summaries for
        softmaxes histograms.
    """
    tensors_type_softmax = [ [tf.placeholder(tf.float32, shape = [N_TYPES,], name = "type_softmax_subpol{}_op{}".format(i, j)) for j in range(N_OPS)] for i in range(N_SUBPOL) ]
    tensors_prob_softmax = [ [tf.placeholder(tf.float32, shape = [N_PROBS,], name = "prob_softmax_subpol{}_op{}".format(i, j)) for j in range(N_OPS)] for i in range(N_SUBPOL) ]
    tensors_magn_softmax = [ [tf.placeholder(tf.float32, shape = [N_MAG,], name = "magn_softmax_subpol{}_op{}".format(i, j)) for j in range(N_OPS)] for i in range(N_SUBPOL) ]

    type_summaries = [ [ tf.summary.histogram(tensor.name, tensor) for tensor in op_tensors ] for op_tensors in tensors_type_softmax]
    prob_summaries = [ [ tf.summary.histogram(tensor.name, tensor) for tensor in op_tensors ] for op_tensors in tensors_prob_softmax]
    magn_summaries = [ [ tf.summary.histogram(tensor.name, tensor) for tensor in op_tensors ] for op_tensors in tensors_magn_softmax]

    return tf.summary.merge([type_summaries, prob_summaries, magn_summaries]), [tensors_type_softmax, tensors_prob_softmax, tensors_magn_softmax]

def feed_histogram_summaries(tensors_type_softmax, tensors_prob_softmax, tensors_magn_softmax, type_softmax, prob_softmax, magn_softmax):
    """
        This function creates the feed dict for softmaxes histograms.
    """

    feed_type = dict([(tensors_type_softmax[i][j], type_softmax[j][0, :]) for i in range(N_SUBPOL) for j in range(N_OPS)])
    feed_prob = dict([(tensors_prob_softmax[i][j], prob_softmax[j][0, :]) for i in range(N_SUBPOL) for j in range(N_OPS)])
    feed_magn = dict([(tensors_magn_softmax[i][j], magn_softmax[j][0, :]) for i in range(N_SUBPOL) for j in range(N_OPS)])

    return {**feed_type, **feed_prob, **feed_magn}

def gradients_summaries(model):
    """
        Creates histograms for model gradients
    """
    l2_norm = lambda t : tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
    g_histo = []
    v_histo = []

    softmaxes_old = [tf.placeholder(tf.float32, shape = output.shape) for output in model.outputs]
    advantage     = tf.placeholder(tf.float32, shape = ())
    loss_func     = proximal_policy_gradient_loss(model.outputs, softmaxes_old, advantage)
    grads         = tf.gradients(loss_func, model.trainable_weights)
    grads         = zip(grads, model.trainable_weights)

    for g, v in grads:
        g_histo.append(tf.summary.histogram("gradients/" + v.name, l2_norm(g)))
        v_histo.append(tf.summary.histogram("variables/" + v.name, l2_norm(v)))

    return tf.summary.merge([*g_histo, *v_histo])

def proximal_policy_gradient_loss(softmaxes, softmaxes_old, advantage, epsilon = 0.2):
    rt      = tf.reduce_mean([K.mean(softmax / softmax_old) for softmax, softmax_old in zip(softmaxes, softmaxes_old)])
    rt_clip = K.clip(rt, 1 - epsilon, 1 + epsilon)
    loss    = K.minimum(rt * advantage, rt_clip * advantage)

    return loss

def get_baseline(mem_rewards, len_avg = 3):
    if len(mem_rewards) < len_avg:
        return mem_rewards[-1]
    else:
        return sum(mem_rewards[-len_avg:]) / len_avg

class Controller:
    def __init__(self, tf_session, dataset_obj, saver, load_pretrain = True, image_shape = [28, 28, 1], p_greedy = 0.2, n_epochs = 5000):
        self.dataset     = dataset_obj       # Dataset API obj
        self.session     = tf_session        # Tensorflow session
        self.saver       = saver             # Policy/model saver 
        self.image_shape = image_shape       # Input image shape
        self.p_greedy    = p_greedy          # Probability of acting greedy
        self.n_epochs    = n_epochs          # Number of control train epochs
        self.model = self.build_controller() # Controller model

        self.softmaxes_old = [tf.placeholder(tf.float32, shape = output.shape) for output in self.model.outputs]
        self.advantage     = tf.placeholder(tf.float32, shape = ())
        self.loss_func     = proximal_policy_gradient_loss(self.model.outputs, self.softmaxes_old, self.advantage)
        self.grad_hist     = gradients_summaries(self.model)
        self.grads         = tf.gradients(self.loss_func, self.model.trainable_weights)
        self.grads         = [(- 1) * g for g in self.grads]
        self.grads         = zip(self.grads, self.model.trainable_weights)

        self.optmizer = tf.train.GradientDescentOptimizer(0.001).apply_gradients(self.grads)

    def build_controller(self):
        """
        Builds controller computational graph
        """
        with tf.variable_scope("Controller"):
            input_layer = layers.Input(shape = INPUT_SHAPE)
            initializer = initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

            input_layers   = [input_layer]
            hidden_layers  = []
            output_softmaxes = []

            for i in range(N_SUBPOL):
                hidden_layers.append(layers.CuDNNLSTM(units = N_UNITS, kernel_initializer = initializer)(input_layers[-1]))
                output_layer = []
                for j in range(N_OPS):
                    name = "subpol_{}_operation_{}".format(i + 1, j + 1)
                    output_layer.extend([
                        layers.Dense(N_TYPES, activation ='softmax', name = name + '_type', kernel_initializer = initializer)(hidden_layers[-1]),
                        layers.Dense(N_PROBS, activation ='softmax', name = name + '_prob', kernel_initializer = initializer)(hidden_layers[-1]),
                        layers.Dense(N_MAG, activation ='softmax', name = name + '_magn', kernel_initializer = initializer)(hidden_layers[-1])
                    ])

                output_softmaxes.append(output_layer)
                input_layers.append(layers.Lambda(expand_dims)(layers.Concatenate()(output_layer)))
            output_list = [item for sublist in output_softmaxes for item in sublist]
            model = models.Model(input_layer, output_list)
        exists = os.path.isfile(os.path.join(LOG_DIR, "controller_model", "model.json"))
        if not exists:
            model_json = model.to_json() # Converts model to JSON
            with open(os.path.join(LOG_DIR, "controller_model", "model.json"), "w") as json_file:
                json_file.write(model_json) # Write to file

        return model
    
    def fit(self, mem_softmax, mem_advantage, epoch):
        """
        Perform a train controller train step
        """

        # Averaing over past history:

        # Reward normalization factor
        min_advantage = np.min(mem_advantage)
        max_advantage = np.max(mem_advantage)

        controller_loss = 0

        for old_softmaxes, softmaxes, advantage in zip(mem_softmax[-2::-1], mem_softmax[::-1], mem_advantage[::-1]):
            initial_input = np.expand_dims(np.concatenate(softmaxes[-6:], axis = 1), axis = -1) # At this time-step, initial input
            # was the last six softmaxes, that is, the softmax output at the fifth LSTM layer.

            # Dictionaries:
            # Dictionary of inputs
            dict_inputs  = {self.model.input : initial_input}   
            # Dictionary of old softmaxes: picks each softmaxes (of a total of 30) and matches it with its tensor
            dict_old     = {old_softmax : s for old_softmax, s in zip(self.softmaxes_old, old_softmaxes)}
            # Advantage dictionary. Picks the corresponding advantage and normalizes it.
            dict_adv     = {self.advantage: (advantage - min_advantage) / (max_advantage - min_advantage)}
            # Dictionary of outputs.
            dict_outputs = {_output : s for _output, s in zip(self.model.outputs, softmaxes)}
            # Feed dict for loss optimization
            feed_dict = {**dict_outputs, **dict_adv, **dict_old, **dict_inputs}

            controller_loss = controller_loss + self.session.run(self.loss_func, feed_dict = feed_dict)
            self.session.run(self.optmizer, feed_dict = feed_dict)

        return controller_loss / len(mem_advantage)

    def sample_policy(self, initial_input):
        """
        Predict softmaxes using 
        """
        initial_input = np.zeros([1, *INPUT_SHAPE]) # START input
        softmaxes     = self.model.predict(initial_input) # Softmax prediction
        
        # Converts softmaxes into subpolicies
        subpolicies   = []
        for i in range(N_SUBPOL):
            operations = []
            for j in range(N_OPS):
                k = (i * N_OPS) + j
                op_softmax = softmaxes[3 * k : 3 * (k + 1)]   # For each k, k: type softmax, k+1: prob softmax, k+2: mag softmax
                op_params  = [o[0, :] for o in op_softmax] # The procedure is done for each subpolicy i
                if np.random.rand() < self.p_greedy:
                    operations.append(Operation(*op_params, greedy = True))
                else:
                    operations.append(Operation(*op_params))
            subpolicies.append(Subpolicy(*operations))

        return softmaxes, subpolicies

    def train(self):
        # Placeholders for Tensorboard
        adv_tensor = tf.placeholder(tf.float32, shape = [], name = "Advantage")
        test_rmse  = tf.placeholder(tf.float32, shape = [], name = "Test_RMSE")
        test_ssim  = tf.placeholder(tf.float32, shape = [], name = "Test_SSIM")
        test_psnr  = tf.placeholder(tf.float32, shape = [], name = "Test_psnr")
        closs      = tf.placeholder(tf.float32, shape = [], name = "Controller_Loss")
        im_result  = tf.placeholder(tf.uint8, shape = [1, 8 * self.image_shape[0], 3 * self.image_shape[1], self.image_shape[2]], name = "Input_Placeholder")

        # Histogram summaries
        softmax_histograms, tensors = softmax_summaries()

        # Summaries
        summ_rmse        = tf.summary.scalar('RMSE', test_rmse)
        summ_ssim        = tf.summary.scalar('SSIM', test_ssim)
        summ_psnr        = tf.summary.scalar('PSNR', test_psnr)
        summ_rew         = tf.summary.scalar('Advantage', adv_tensor)
        summ_closs       = tf.summary.scalar('Controller_Loss', closs)
        summ_denoising   = tf.summary.image('Denoising_Results', im_result)
        merged           = tf.summary.merge([summ_rmse, summ_psnr, summ_ssim, summ_rew])
        tf.global_variables_initializer().run(session = self.session)
        self.writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "training_monitor"), graph = self.session.graph)
        
        # Learning loop
        mem_softmaxes = [] # Softmax memory
        mem_rewards   = [] # Reward memory
        mem_advantage = [] # Advantage memory
        old_softmax   = np.zeros([1, *INPUT_SHAPE])
        print("Starting Controller training")
        for epoch in range(self.n_epochs):
            start_epoch = time.time()
            print("Epoch\t{}/{}".format(epoch, self.n_epochs)  + 149 * "-")
            # Policy prediction
            softmaxes, subpolicies = self.sample_policy(old_softmax)

            # Updates old softmax
            old_softmax = np.expand_dims(np.concatenate(softmaxes[-6:], axis = 1), axis = -1)

            # Softmax memory
            mem_softmaxes.append(softmaxes)

            # Creates new child and trains it
            child = Autoencoder(self.dataset, n_epochs = 250, n_stages = 20, test_runs = 50, lr = 0.02, decay = 0.1)
            train_start = time.time()
            child.train(policy = subpolicies, do_valid = False)
            elapsed = time.time() - train_start
            print("Training time:\t{}".format(timedelta(seconds = elapsed)))

            # Logging training data into tensorboard
            train_start = time.time()
            loss, ssim, rmse, psnr = child.test() # Evaluates training results
            elapsed = time.time() - train_start
            reward    = psnr                      # Epoch reward
            mem_rewards.append(reward)            # Keeps a memory of rewards
            baseline  = get_baseline(mem_rewards) # Simple moving average
            advantage = reward - baseline         # Computes advantage
            mem_advantage.append(advantage)       # Keeps a memory of advantages
            print("Eval time:\t{}".format(timedelta(seconds = elapsed)))

            # Saving child model
            self.saver(child.model, subpolicies, mem_rewards[-1])

            # Weight update
            if len(mem_softmaxes) > 5:
                # Perform backprop only if we have enough softmaxes in memory
                controller_loss = self.fit(mem_softmaxes, mem_advantage, epoch)
                self.model.save_weights(os.path.join(LOG_DIR, "controller_model/model.h5"))
                closs_summary = self.session.run(summ_closs, feed_dict = {closs : controller_loss})
                self.writer.add_summary(closs_summary, epoch)

            # Logging data into tensorboard

            # Scalar summaries
            feed_dict = {
                test_rmse  : rmse,
                test_ssim  : 1 - ssim,
                test_psnr  : psnr,
                adv_tensor : advantage
            }

            summ = self.session.run(merged, feed_dict = feed_dict)
            self.writer.add_summary(summ, epoch)

            print("SSIM: \t\t{}".format(1 - ssim))
            print("RMSE: \t\t{}".format(rmse))
            print("PSNR: \t\t{}".format(psnr))
            print("Reward: \t{}".format(mem_rewards[-1]))
            # Images and histograms
            if epoch % 10 == 0:
                # Image summary
                x, y, r = child.inference() 
                
                # Converts 8 first samples to uint8
                noise    = np.uint8(255 * x[0:8]) 
                original = np.uint8(255 * y[0:8])
                denoised = np.uint8(255 * r[0:8])

                # Stacking images per batch
                a = np.vstack([noise[i]    for i in range(len(noise))])
                b = np.vstack([original[i] for i in range(len(original))])
                c = np.vstack([denoised[i] for i in range(len(denoised))])

                # Stacking batches side-by-side
                img = np.expand_dims(np.hstack([a, b, c]), 0)

                img_summ = self.session.run(summ_denoising, feed_dict = {im_result: img})
                self.writer.add_summary(img_summ, epoch)
                # Histograms
                type_softmaxes = [softmaxes[0], softmaxes[3]]
                prob_softmaxes = [softmaxes[1], softmaxes[4]]
                magn_softmaxes = [softmaxes[2], softmaxes[5]]
                feed_hist = feed_histogram_summaries(*tensors, type_softmaxes, prob_softmaxes, magn_softmaxes)
                
                soft_summ = self.session.run(softmax_histograms, feed_dict = feed_hist)
                self.writer.add_summary(soft_summ, epoch)
            print("Epoch took:\t{}".format(timedelta(seconds = time.time() - start_epoch)))