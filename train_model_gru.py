from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, concatenate, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import LSTM, Bidirectional, GRU
import tensorflow as tf
from tensorflow.python.keras import backend as K

from metrics import keras_ndcg
from set_variable import *

class MyModel:

    def __init__(self):
        self._create()

    def _create(self):

        EMBEDDING_DIMS = 100
        GRU_DIMS = 64
        DROPOUT_GRU = 0.2
        RNN_DROPOUT = 0.2
        print('Creating Model...')
        # EMBEDDING
        input_play = Input(shape=(SEQ_LEN,), dtype='int32', name='input_play')
        input_save = Input(shape=(SEQ_LEN,), dtype='int32', name='input_save')

        # Keras requires the total_dim to have 2 more dimension for "other" class
        embedding_layer = Embedding(
                input_dim=(EMBEDDING_CLASSES + 2), output_dim=EMBEDDING_DIMS,
                input_length=SEQ_LEN, mask_zero=True, trainable=True,
                name='emb')

        input_play_encoded = embedding_layer(input_play)
        input_save_encoded = embedding_layer(input_save)

        play_gru = GRU(GRU_DIMS, return_sequences=False, dropout=DROPOUT_GRU,
                       recurrent_dropout=RNN_DROPOUT, name='gru1')(input_play_encoded)
        save_gru = GRU(GRU_DIMS, return_sequences=False, dropout=DROPOUT_GRU,
                       recurrent_dropout=RNN_DROPOUT, name='gru2')(input_save_encoded)

        # TIME_OF_DAY OHE
        tod_inp = Input(shape=(TOTAL_TOD_BINS,), name='time_of_day_ohe')

        # DAY_OF_WEEK OHE
        dow_inp = Input(shape=(TOTAL_DOW_BINS,), name='day_of_wk_ohe')

        # MERGE LAYERS
        print('Merging features...')
        merged = concatenate(
            [play_gru, save_gru, tod_inp, dow_inp], axis=1,
            name='concat')

        # FULLY CONNECTED LAYERS
        dense = Dense(1024, activation='relu', name='main_dense')(merged)
        pred = Dense(TARGET_CLASSES, activation='softmax', name='output')(dense)

        self.model = Model(
            inputs=[input_play, input_save, tod_inp, dow_inp],
            outputs=[pred])

        print(self.model.summary())

        return self.model

    def compile(self):

        self.model.compile(
            optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
            metrics=[keras_ndcg(k=5)])
