# %% [code]
# %% [code]
import numpy as np

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.signal  import decimate
import pandas as pd
import gc
import tensorflow as tf



EPOCH_LENGTH = 30

SAMPLE_RATE = 250
SAMPLE_RATE_AIRFLOW = 10
SAMPLE_RATE_BODYPOSITION = 10
OUT_DIM = 4  # len(idDict)
BATCH_SIZE = 4
TEST_SIZE = 32
ECG_TIME_STEPS = SAMPLE_RATE * EPOCH_LENGTH
AIR_TIME_STEPS = SAMPLE_RATE_AIRFLOW * EPOCH_LENGTH
BP_TIME_STEPS = SAMPLE_RATE_BODYPOSITION * EPOCH_LENGTH
STEP = 1
STAGES = 4


# credits: https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, list_IDs, ecg_path, airflow_path, bp_path, hypnogram_path, healthy_path,
                 to_fit=True, batch_size=32, shuffle=True, weights= True, class_weights = None, class_weights_status = None):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.hypnogram_path = hypnogram_path
        self.airflow_path = airflow_path
        self.healthy_path = healthy_path
        self.ecg_path = ecg_path
        self.bp_path = bp_path
        self.list_IDs = list_IDs
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weights = weights
        self.class_weights_status = class_weights_status
        self.class_weights = class_weights
        self.on_epoch_end()
        
        
        healthy = np.loadtxt(healthy_path + 'apnea_ahi_a0h3.csv',delimiter = ',')
        
        #print(healthy.shape)
        
        self.ids = healthy[:,0]
        self.healthy = healthy[:,1]
        
        del healthy
        

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        
        gc.collect()
        
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X_ecg, X_af, X_bp= self._generate_X(list_IDs_temp)

        if self.to_fit:
            y, h = self._generate_y(list_IDs_temp)

            if self.weights:
                w = np.take(self.class_weights, y[:, :, 0])
                #print(w.shape)
                #print(str(y[0][0])+'--------'+str(w[0][0]))
                #print(y[0].shape)
                
                w2 = np.take(self.class_weights_status, h[:,:,0])
                
                return [X_ecg, X_af,X_bp], [y,h] , [w,w2]
            else:
                return [X_ecg, X_af,X_bp], [y,h]
        else:
            return [X_ecg, X_af, X_bp]

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X_ecg = []
        X_af = []
        X_bp = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            ecg_temp, af_temp, bp_temp = self._load_ecg_airflow_bp(self.ecg_path, self.airflow_path,self.bp_path, ID)
            
            #ecg_temp = decimate(ecg_temp,2) #down-sampling at 125hz
            X_ecg.append(ecg_temp)
            X_af.append(af_temp)
            X_bp.append(bp_temp)
            
            del ecg_temp
            del af_temp
            del bp_temp



        #print(X_ecg.shape)
        X_ecg = pad_sequences(X_ecg,value=0,padding='post')
        #print(X_ecg.shape)
        
        

        X_ecg = X_ecg.reshape([X_ecg.shape[0],int(len(X_ecg[0]) / ECG_TIME_STEPS), ECG_TIME_STEPS, 1, 1])
        #print(X_ecg.shape)

        #print(X_af.shape)
        X_af = pad_sequences(X_af,value=0,padding='post')
        #print(X_af.shape)

        X_af = X_af.reshape([X_af.shape[0],int(len(X_af[0]) / AIR_TIME_STEPS), AIR_TIME_STEPS, 1, 1])
        #print(X_af.shape)
        
        X_bp = pad_sequences(X_bp,value=0,padding='post')
        #print(X_bp.shape)

        X_bp = X_bp.reshape([X_bp.shape[0],int(len(X_bp[0]) / BP_TIME_STEPS), BP_TIME_STEPS, 4, 1])
        #print(X_bp.shape)



        X_ecg = X_ecg.astype(np.float32)    
        X_af = X_af.astype(np.float32)
        X_bp = X_bp.astype(np.float32)



        return X_ecg, X_af, X_bp

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = []
        
        h = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y.append(self._load_hypnogram(self.hypnogram_path, ID))
            h.append(self._load_healthy(self.healthy_path,ID,))

        y = pad_sequences(y,value=0,padding='post')
        
        #ln = len(y[0])

        #y = y.reshape(y.shape[0],y.shape[1],y.shape[2],1)

        #y = y.astype(np.int32)
        """
        y_padded = self._padding_batches(y, target=True)
        del y
        y_padded_casted = y_padded.astype(np.int32)
        """
        
        h = np.asarray(h)
        h = h.astype(np.int32)

        return y, h

    def _load_ecg_airflow_bp(self, ecg_path, airflow_path,bp_path, id):
        """Load grayscale image
        :return: loaded image
        """
        name = id + '.npz'


        ecg = np.load(ecg_path + name)
        ecg = ecg['arr_0']

        #ecg_reshaped = ecg.reshape([int(len(ecg) / (EPOCH_LENGTH * SAMPLE_RATE)), int(EPOCH_LENGTH * SAMPLE_RATE), 1, 1])
        #del ecg

        af = np.load(airflow_path + name)
        af = af['arr_0']

        #af_reshaped = af.reshape(
        #    [int(len(af) / (EPOCH_LENGTH * SAMPLE_RATE_AIRFLOW)), int(EPOCH_LENGTH * SAMPLE_RATE_AIRFLOW), 1, 1])
        #del af
        
        bp = np.load(bp_path + name)
        bp = bp['arr_0']

        return ecg, af, bp

    def _load_hypnogram(self, hypnogram_path, id):
        """
        Load hypnogram
        :param hypnogram_path: path to hypnogram fodler
        :param id: identifier
        :return: hypnogram
        """
        hypnogram_name = id + '.csv'

        hypnogram = pd.read_csv(hypnogram_path + hypnogram_name, usecols=['Stage'])
        #hypnogram.rename(columns={'Sleep': 'Y'}, inplace=True)
        
        hynpgram_reshaped = np.array(hypnogram).reshape(-1, 1)
        del hypnogram

        return hynpgram_reshaped
    
    
    def _load_healthy(self, healthy_path, id):
    
        
        index = np.where(self.ids == int(id))
        
        #print(str(index))
        
        return [self.healthy[index]]
        
        

#    def _padding_batches(self, v=None, fillval=0, target=False, time_steps=7500):
#        """ Padd the batch in order to have all singla with same length
#        :param v: list of samples
#        :param fillval: value to use to padding
#        :param target: True if v is the target
#        :param time_steps: sample_rate * epoch length
#        :return: batch
#        """
#        lens = np.array([len(item) for item in v])
#        mask = lens[:, None] > np.arange(lens.max())
#
#        del lens
#        # print("mask shape: ")
#        # print(mask.shape)
#        if target:
#            out = np.full((mask.shape[0], mask.shape[1], 1), fillval)
#        else:
#            out = np.full((mask.shape[0], mask.shape[1], time_steps, 1, 1), fillval)
#
#        # print("out shape: ")
#        # print(out.shape)
#
#        out[mask] = np.concatenate(v)
#
#        del v
#        del mask
#
#        return out
#
