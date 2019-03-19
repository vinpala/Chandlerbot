#################################################################################3
#To speed up training -

#1) replace standard LSTM to CuDNNLSTM.
#  Cudnn LSTM is absurdly faster the the "regular" version(10X).

#2) Increase batch_size : 4 to 32

#3) Increase learning rate : 0.0001 to 0.01
#
#################################################################################
import numpy as np
import keras
import pickle
from keras import backend as K
import os
from os import environ
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import json
from emetrics import EMetrics
import models

###############################################################################
# Set up working directories for data, model and logs.
###############################################################################
model_filename = "baseline_bidirec.h5"

#Writing the train model and getting input data

#Output your trained model to a folder named 'model' under the folder specified in the environment variable RESULT_DIR.
#When you store a training-run into the repository, this 'model' folder is saved.

if environ.get('RESULT_DIR') is not None:
    output_model_folder = os.path.join(os.environ["RESULT_DIR"], "model")
    output_model_path = os.path.join(output_model_folder, model_filename)
else:
    output_model_folder = "model"
    output_model_path = os.path.join("model", model_filename)

os.makedirs(output_model_folder, exist_ok=True)

#Input data files located in the specified COS bucket are available to
#your program via the folder specified in environment variable DATA_DIR

input_data_folder = os.environ["DATA_DIR"]

#writing metrics
# create TensorBoard instance for writing training metrics
log_dir = os.environ.get("LOG_DIR")
sub_id_dir = os.environ.get("SUBID")
static_path_train = os.path.join("logs", "tb", "train")
if log_dir is not None and sub_id_dir is not None:
    tb_directory_train = os.path.join(log_dir, sub_id_dir, static_path_train)
    tensorboard = TensorBoard(log_dir=tb_directory_train)
else:
    tb_directory_train = static_path_train
    tensorboard = TensorBoard(log_dir=tb_directory_train)
###############################################################################

###############################################################################
# Set up HPO.
###############################################################################

config_file = "config.json"

if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        json_obj = json.load(f)
    learning_rate = json_obj["learning_rate"]
    hidden_size = json_obj["hidden_size"]
    batch_size = json_obj["batch_size"]
    num_epochs = json_obj["num_epochs"]
else:
    learning_rate = 0.001
    #hidden_size = 500
    #batch_size = 64
    batch_size = 32 #to tackle memory error
    num_epochs = 10

def getCurrentSubID():
    if "SUBID" in os.environ:
        return os.environ["SUBID"]
    else:
        return None

class HPOMetrics(keras.callbacks.Callback):
    def __init__(self):
        self.emetrics = EMetrics.open(getCurrentSubID())

    def on_epoch_end(self, epoch, logs={}):
        train_results = {}

        print('EPOCH ' + str(epoch))
        self.emetrics.record("train", epoch, train_results)

    def close(self):
        self.emetrics.close()
###############################################################################

###############################################################################
# Load and prepare data
###############################################################################
# Load data from pickle object
import pickle

X_train = pickle.load(open(os.path.join(input_data_folder,'X_train.pickle'), 'rb'))
y_train = pickle.load(open(os.path.join(input_data_folder,'y_train.pickle'), 'rb'))
X_val = pickle.load(open(os.path.join(input_data_folder,'X_val.pickle'), 'rb'))
y_val = pickle.load(open(os.path.join(input_data_folder,'y_val.pickle'), 'rb'))
X_word_to_idx = np.load(os.path.join(input_data_folder,'X_word_to_idx.npy'))
X_idx_to_word = np.load(os.path.join(input_data_folder,'X_idx_to_word.npy'))
y_word_to_idx = np.load(os.path.join(input_data_folder,'y_word_to_idx.npy'))
y_idx_to_word = np.load(os.path.join(input_data_folder,'y_idx_to_word.npy'))
parameters    = np.load(os.path.join(input_data_folder,'parameters.npy'))
X_vocab_size = len(X_word_to_idx.item())
y_vocab_size = len(y_word_to_idx.item())
X_max_len =  parameters.item()['X_max_len']
y_max_len =  parameters.item()[' y_max_len']
#vectorize (or one-hot-encode) y_train only, for X_train we will create an embedding

################################################################################
# Batch Generator
################################################################################
#This implementation generates the dataset on multiple cores in real time and feed it right away to the deep learning model.
#Useful for Large datasets ref : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#Sequence are a safer way to do multiprocessing.
#This structure guarantees that the network will only train once on each sample per epoch which is not the case with generators.
#The method __getitem__ should return a complete batch
class BatchGenerator(keras.utils.Sequence):
    def __init__(self, X, y, X_max_len, y_max_len, batch_size, vocabulary, shuffle=True):
        'Initialization'
        self.X = X
        self.y = y
        self.X_max_len = X_max_len
        self.y_max_len = y_max_len
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X))
        #print(int(np.ceil(len(self.X)/self.batch_size)))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.X)/self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X_batch, y_batch = self.__data_generation(indexes)

        #self.display_batch(X_batch,y_batch)
        return X_batch, y_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        X_batch = np.zeros((self.batch_size, self.X_max_len))
        y_batch = np.zeros((self.batch_size, self.y_max_len, self.vocabulary))

        # Generate data
        for i, index in enumerate(indexes):
            # Utterance
            X_batch[i,:] = self.X[index]

            # Response
            y_batch[i,:,:] = to_categorical(self.y[index], num_classes=self.vocabulary)

        return X_batch, y_batch

    def display_batch(self, X_batch, y_batch):
        for U_sample,R_sample in zip(X_batch, y_batch):
            #print(U_sample)
            #print(R_sample)
            R_sample = np.argmax(R_sample, axis=1)
            print("U : "+' '.join([X_idx_to_word[int(i)] for i in U_sample if i > 0][::-1]))
            print("R : "+' '.join([y_idx_to_word[int(i)] for i in R_sample if i > 0]))
#######################################################################################
#Baseline Model parameters
learning_rate = 0.001
encoder_units = 100
encoder_embed_dim = 100
encoder_layers = 2
decoder_units = 100
decoder_embed_dim = 0
decoder_layers =3

#batch_size = 4
#num_epochs = 10


#model = models.encoder_decoder(encoder_units,encoder_embed_dim,encoder_layers,
                         #decoder_units,decoder_embed_dim,decoder_layers,
                         #X_vocab_size,y_vocab_size,X_max_len, y_max_len,learning_rate)
# Adding gradient-clipping and and regularization because the training was frozen after
#9589/13838 [===================>..........] - ETA: 1:33:37 - loss: 0.6424 - acc: 0.9302
model = models.encoder_decoder(encoder_units,encoder_embed_dim,encoder_layers,
                         decoder_units,decoder_embed_dim,decoder_layers,
                         X_vocab_size,y_vocab_size,X_max_len, y_max_len,learning_rate,clipvalue=0.5,weight_decay=0.000001,
                         bidirectional=True)
hpo = HPOMetrics()

train_data_generator = BatchGenerator(X_train, y_train, X_max_len, y_max_len, batch_size, y_vocab_size)

history = model.fit_generator(generator=train_data_generator,epochs=num_epochs,verbose=1)
                    #use_multiprocessing=True)
                    #this throws a weird error : multiprocessing.pool.MaybeEncodingError
                    #Tensorflow already uses some kind of multiprocessing to efficiently run computation

print("Training history:" + str(history.history))

hpo.close()

# save the model
model.save(output_model_path)
# save the model
#test_model.save(output_model_path)
