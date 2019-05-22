from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



def vae_model_single(path,original_dim,xtrain,xtest,intermediate_dim,batch_size,latent_dim,epochs):
    # network parameters
    input_shape = (original_dim, )
    intermediate_dim = intermediate_dim
    batch_size = batch_size
    latent_dim = latent_dim
    epochs = epochs

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    #x = Dense(intermediate_dim, name='encoder_h1', activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(inputs)
    z_log_var = Dense(latent_dim, name='z_log_var')(inputs)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, z_mean, name='encoder')
    encoder.summary()
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    #x = Dense(intermediate_dim, name='decoder_h1', activation='relu')(latent_inputs)
    outputs = Dense(original_dim, name='decoder_mu', activation='sigmoid')(latent_inputs)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    

    # instantiate VAE model
    outputs = decoder(z)
    vae = Model(inputs, outputs, name='vae_mlp')
    '''
    def vae_loss(inputs, outputs):
        xent_loss = mse(inputs, outputs)
        #xent_loss = binary_crossentropy(inputs, outputs)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
    '''
    #reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss = binary_crossentropy(inputs,outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)  
      
        
    vae.add_loss(vae_loss)
    
    vae.compile(optimizer='adadelta', loss=None)
    
    vae.summary()
    history=vae.fit(xtrain, None, epochs=epochs, batch_size=batch_size, validation_data=(xtest, None))
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(path, "vae_training_history.csv"))
    encoder.save(os.path.join(path,"vae_encoder.h5"))
    decoder.save(os.path.join(path,"vae_decoder.h5"))
    K.clear_session()
