#References -
#https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
#https://colab.research.google.com/github/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb#scrollTo=s5hQWlbN3jGF
import keras
import numpy as np
from keras.layers import Activation, TimeDistributed, Dense, Input,RepeatVector, recurrent, Embedding, Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import  CuDNNGRU
from keras.layers import Bidirectional
import tensorflow as tf
import random
from keras.regularizers import l2
from keras.initializers import Constant
# seed weight initialization
random.seed(42)
np.random.seed(42)

def gru(units, regularizer, bidirectional, return_sequences):
  # If a GPU is available use CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
    if tf.test.is_gpu_available():
        gru = CuDNNGRU(units, return_sequences=return_sequences,
                                    kernel_regularizer=regularizer,
                                    recurrent_regularizer=regularizer,
                                    bias_regularizer=regularizer,
                                    recurrent_initializer='glorot_uniform')
    else:
        gru = GRU(units, return_sequences=return_sequences,
                                        kernel_regularizer=regularizer,
                                        recurrent_regularizer=regularizer,
                                        bias_regularizer=regularizer,
                                        recurrent_initializer='glorot_uniform')
    if bidirectional:
        return Bidirectional(gru)
    return gru

def encoder_decoder(encoder_units,encoder_embed_dim,encoder_layers,
                         decoder_units,decoder_embed_dim,decoder_layers,
                         X_vocab_size,y_vocab_size,X_max_len, y_max_len,
                         learning_rate,optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'],
                         clipvalue=None,weight_decay=None,bidirectional=False,pretrained_embed=None,dropout=0.5):

    model = Sequential()

    regularizer = l2(weight_decay) if weight_decay else None

    #emb_weights = [pretrained_embed] if pretrained_embed else None

    ############   ENCODER   ##################
    #mask_zero s set to True because Masking is not supported for CuDNN RNNs.
    #For each timestep in the input tensor (dimension #1 in the tensor),
    #if all values in the input tensor at that timestep are equal to mask_value,
    #then the timestep will be masked (skipped) in all downstream layers (as long as they support masking).
    if tf.test.is_gpu_available():
        mask_zero = False
    else:
        mask_zero = True
    #The Embedding layer a trainable look-up table , it is initialized with random weights and will learn an embedding for all of the words in the training dataset.
    #creates a weight matrix of (vocabulary_size)x(embedding_dimension) dimensions
    #hidden-size is the size of the vector space in which words will be embedded.
    #It defines the size of the output vectors from this layer for each word.
    #hidden_size = the latent dimension
    if pretrained_embed is not None:
        embedding_layer = Embedding(X_vocab_size,
                            pretrained_embed.shape[1],
                            embeddings_initializer=Constant(pretrained_embed),
                            input_length=X_max_len,
                            trainable=False)
        model.add(embedding_layer)
    else:
        model.add(Embedding(X_vocab_size, encoder_embed_dim, input_length=X_max_len,
                    embeddings_regularizer=regularizer, weights=None, mask_zero=mask_zero,
                    name='encoder_embedding'))

    if encoder_layers > 1:
        for _ in range(1, encoder_layers):
            model.add(gru(encoder_units, regularizer, bidirectional, return_sequences=True))
            model.add(Dropout(dropout))
    model.add(gru(encoder_units, regularizer, bidirectional, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(RepeatVector(y_max_len))

    ############   DECODER   ##################
    bidirectional = false #Decoder cannot be bidirectional
    #as the decoder produces word after word in the order of the sentence. Thus it doesn't have access to the future of the sentence
    #(and the bidirectional implies using both a left to right and a right to left LSTM).
    for _ in range(1, decoder_layers+1):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
        model.add(gru(decoder_units, regularizer, bidirectional, return_sequences=True))
        model.add(Dropout(dropout))

    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    model.add(TimeDistributed(Dense(y_vocab_size, activation='softmax') ))

    #############################################
    if optimizer == 'adam':
        opt = Adam(lr=learning_rate, clipvalue=clipvalue)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=learning_rate, clipvalue=clipvalue)

    model.compile(loss=loss,
              optimizer=opt,
              metrics=metrics)
    return model
