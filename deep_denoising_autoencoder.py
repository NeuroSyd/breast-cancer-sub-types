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

#return prediction for x_test
def autoencoder_predict(encoder, x_test):
	prediction = encoder.predict(x_test)
	return prediction.reshape((len(prediction), np.prod(prediction.shape[1:])))

#return a fit deep encoder
def deep_denoising_autoencoder_fit(x_train, x_test, x_train_noisy, x_test_noisy, 
        encoding_dim=2, optimizer="adadelta", loss_function="binary_crossentropy", nb_epoch=4, batch_size=2, path='./saved_models'):
        input_img = Input(shape=(x_train.shape[1],), name="x")

        #"encoded" is the encoded representation of the input
    
        #x_train, x_valid = model_selection.train_test_split(x_train, test_size=0.10, random_state=seed)
        encoded = Dense(1000, name="encoder_h1", activation='relu')(input_img)
        encoded = Dense(encoding_dim, name="encoder_mu", activation='relu')(encoded)

        
        #decoded" is the lossy reconstruction of the input
        decoded = Dense(1000, name="decoder_h1", activation='relu')(encoded)
        decoded = Dense(x_train.shape[1], name="decoder_mu", activation='sigmoid')(decoded)
        

        autoencoder = Model(inputs=input_img, outputs=decoded)
        
        encoder = Model(inputs=input_img, outputs=encoded)
        
        autoencoder.compile(optimizer=optimizer, loss=loss_function)

        autoencoder.fit(x_train_noisy, x_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True
                        )
        encoder.save(os.path.join(path, "denoising_encoder.h5"))
        
