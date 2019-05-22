seed=75
import os
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
np.random.seed(seed)
from tensorflow import set_random_seed
set_random_seed(seed)
from keras.layers import Dense, Reshape, Flatten, Input, merge, Dropout, LeakyReLU, Activation
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD, Adagrad, RMSprop, Adadelta
from keras.regularizers import l1, l1_l2
from keras.datasets import mnist
import keras.backend as K
import pandas as pd
#from keras.utils import multi_gpu_model
from keras import backend as K
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, fix_names, n_choice
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling


def model_generator(latent_dim, input_shape, hidden_dim=1000):
    return Sequential([
        Dense(hidden_dim, name="generator_h1", input_dim=latent_dim, activation='tanh'),
        Dense(hidden_dim, name="generator_h2", activation='tanh'),
        Dense(input_shape[0], name="generator_output", activation='sigmoid')],
        name="generator")
        
def model_encoder(latent_dim, input_shape, hidden_dim=1000):
    x = Input(input_shape, name="x")
    h = Dense(hidden_dim, name="encoder_h1", activation='tanh')(x)
    h = Dense(hidden_dim, name="encoder_h2", activation='tanh')(h)
    z = Dense(latent_dim, name="encoder_mu", activation='tanh')(h)
    return Model(x, z, name="encoder")
    
           
def model_discriminator(input_shape, output_dim=1, hidden_dim=1000):
    z = Input(input_shape)
    h = z
    h = Dense(hidden_dim, name="discriminator_h1", activation='tanh')(h)
    h = Dense(hidden_dim, name="discriminator_h2", activation='tanh')(h)
    y = Dense(output_dim, name="discriminator_y", activation="sigmoid")(h)
    return Model(z, y)


def aae_model(path, adversarial_optimizer,xtrain,ytrain,xtest,ytest,encoded_dim=100,img_dim=25, nb_epoch=20):
    # z \in R^100
    latent_dim = encoded_dim
    # x \in R^{28x28}
    input_shape = (img_dim,)

    # generator (z -> x)
    generator = model_generator(latent_dim, input_shape)
    # encoder (x ->z)
    encoder = model_encoder(latent_dim, input_shape)
    # autoencoder (x -> x')
    autoencoder = Model(encoder.inputs, generator(encoder(encoder.inputs)), name="autoencoder")
    # discriminator (z -> y)
    discriminator = model_discriminator(input_shape)

    # assemple AAE
    x = encoder.inputs[0]
    z = encoder(x)
    xpred = generator(z)

    yreal = discriminator(x)
    yfake = discriminator(xpred)
    aae = Model(x, fix_names([xpred, yfake, yreal], ["xpred", "yfake", "yreal"]))

    # print summary of models
    encoder.summary()
    generator.summary()
    
    discriminator.summary()
    #autoencoder.summary()

    # build adversarial model
    generative_params = generator.trainable_weights + encoder.trainable_weights
    model = AdversarialModel(base_model=aae,
                             player_params=[generative_params, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
                             
    #parallel_model = multi_gpu_model(model, gpus=4)
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                             player_optimizers=[Adadelta(),Adadelta()],
                             loss={"yfake": "binary_crossentropy", "yreal": "binary_crossentropy",
                                   "xpred": "binary_crossentropy"},
                                   player_compile_kwargs=[{"loss_weights": {"yfake": 1e-4, "yreal": 1e-4, "xpred": 1e1}}]*2)
    # train network
    n = xtrain.shape[0]
    y = [xtrain, np.ones((n, 1)), np.zeros((n, 1)), xtrain, np.zeros((n, 1)), np.ones((n, 1))]
    ntest = xtest.shape[0]
    ytest = [xtest, np.ones((ntest, 1)), np.zeros((ntest, 1)), xtest, np.zeros((ntest, 1)), np.ones((ntest, 1))]
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest), epochs=nb_epoch, batch_size=128, shuffle=False)
    
    # save history
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(path, "aae_history.csv"))
    # save model
    encoder.save(os.path.join(path, "aae_encoder.h5"))
    generator.save(os.path.join(path, "aae_decoder.h5"))
    discriminator.save(os.path.join(path, "aae_discriminator.h5"))
    K.clear_session()
