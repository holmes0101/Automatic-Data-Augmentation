import keras

from models import SSIM, RMSE, PSNR, TensorBoardImage

class Autoencoder:
    def __init__(self, dataset_obj, n_epochs = 100, n_stages = 100, test_runs = 100, decay = 0.0, lr = 1e-2, emb_size = 7, model_name = "Autoencoder"):
        ## Model parameters
        self.dataset  = dataset_obj             # Dataset object
        self.n_epochs = n_epochs                # Number of training epochs
        self.n_stages = n_stages                # Number of stages for each epoch
        self.test_runs = test_runs              # Number of runs to evaluate test performance
        self.lr = lr                            # Training learning rate
        self.emb_size = emb_size                # Embedding space dimension
        self.decay = decay                      # Optimizer lr decay
        self.model_name = model_name            # Model name
        self.build_model()                      # Creates Keras model

    def build_model(self):
        x = keras.layers.Input(shape = [28, 28, 1])
        # Encoder network
        enc_conv1 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same')(x)
        enc_pool1 = keras.layers.MaxPooling2D(padding = 'same')(enc_conv1)
        enc_conv2 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same')(enc_pool1)
        enc_pool2 = keras.layers.MaxPool2D(padding = 'same')(enc_conv2)
        enc_conv3 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same')(enc_pool2)
        embedding = keras.layers.MaxPool2D(padding = 'same')(enc_conv3)

        # Decoder network
        dec_conv1 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same')(embedding)
        dec_upsa1 = keras.layers.UpSampling2D()(dec_conv1)
        dec_conv2 = keras.layers.Conv2D(8, 3, activation='relu', padding='same')(dec_upsa1)
        dec_upsa2 = keras.layers.UpSampling2D()(dec_conv2)
        dec_conv3 = keras.layers.Conv2D(16, 3, activation='relu')(dec_upsa2)
        dec_upsa3 = keras.layers.UpSampling2D()(dec_conv3)
        z = keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(dec_upsa3) 

        # Model definition
        self.model = keras.models.Model(x, z)
        #adam_opt = keras.optimizers.Adam(lr = self.lr, decay = self.decay)
        self.model.compile(loss = 'binary_crossentropy', optimizer = "adam", 
                           metrics = [SSIM, RMSE, PSNR])



    def train(self, policy = None, do_valid = True):
        """
        Trains the model on a given data batch
        """ 
        def batch_generator(data = "Train", policy = None):
            while True:
                x, y = self.dataset.get_batch(data = data, policy = policy)
                yield x, y
        
        gen_train  = batch_generator(data = "Train", policy = policy)

        if do_valid:
            gen_valid  = batch_generator(data = "Valid", policy = None) 
            ImageCallback = TensorBoardImage(self) # Tensorboard callback
            callbacks = [ImageCallback]
            validation_steps = self.test_runs
        
        else:
            gen_valid = None
            callbacks = None
            validation_steps = None
        
        self.model.fit_generator(
            generator           = gen_train,
            steps_per_epoch     = self.n_stages,
            epochs              = self.n_epochs, 
            validation_data     = gen_valid,
            validation_steps    = validation_steps,
            use_multiprocessing = False, 
            callbacks           = callbacks,
            verbose             = 1
        )

        return self

    def inference(self):
        """
        Denoises a batch of images
        """
        x_batch, y_batch = self.dataset.get_batch(data = "Valid")
        x_denoised = self.model.predict(x_batch)

        return x_batch, y_batch, x_denoised

    def test(self):
        """
        Evaluates the model test_runs times. Returns the average score
        """
        Xtest = self.dataset.x_test
        Ytest = self.dataset.y_test
        
        return self.model.evaluate(x = Xtest, y = Ytest, batch_size = 32, verbose = 0)


if __name__ == "__main__":
    import sys
    import json

    sys.path.append("../")

    from data import MNIST
    from transformations import json2policy

    policy_files = [None, 'model_0_policy.json', 'model_1_policy.json', 'model_2_policy.json']
    policy_filename = policy_files[3]

    
    with open(policy_filename, "r") as f:
        policy_json = json.load(f)
    policy = json2policy(policy_json)


    n_epochs = 250
    lr = 0.02
    decay  = 0.1

    print("Training with policy {}".format(policy_filename))
    print("** Number of Epochs:\t{}".format(n_epochs))

    dataset = MNIST(batch_size = 256, n_train_samples = 500)
    ae = Autoencoder(dataset, n_epochs = n_epochs, n_stages = 20, lr = lr, decay = decay, model_name = "Autoencoder_nopol_500")
    ae.train(do_valid = True, policy = None)
