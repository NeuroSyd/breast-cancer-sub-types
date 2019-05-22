
seed=75
import numpy as np
np.random.seed(seed)
from tensorflow import set_random_seed
set_random_seed(seed)
from sklearn import model_selection
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
from keras import callbacks
import os

def shallow_autoencoder_fit(x_train, x_test, encoding_dim=2, optimizer="adadelta", loss_function="binary_crossentropy", nb_epoch=4, batch_size=2, path='./saved_models'):
  input_img = Input(shape=(x_train.shape[1],), name="x")
  #Shallow Autoencoder
  # "encoded" is the encoded representation of the input
  encoded = Dense(encoding_dim, activation='relu')(input_img)
  # "decoded" is the lossy reconstruction of the input
  decoded = Dense(x_train.shape[1], activation='sigmoid')(encoded)
  # this model maps an input to its reconstruction
  autoencoder = Model(input_img, decoded)
  # this model maps an input to its encoded representation
  encoder = Model(input_img, encoded)
  # create a placeholder for an encoded (32-dimensional) input
  encoded_input = Input(shape=(encoding_dim,))
  # retrieve the last layer of the autoencoder model
  decoder_layer = autoencoder.layers[-1]
  # create the decoder model
  decoder = Model(encoded_input, decoder_layer(encoded_input))
  autoencoder.compile(optimizer='adadelta', loss=loss_function)

  autoencoder.fit(x_train, x_train,
                  epochs=nb_epoch,
                  batch_size=batch_size,
                  shuffle=True,
                  validation_data=(x_test, x_test),
                  )
  encoder.save(os.path.join(path, "shallow_encoder.h5"))