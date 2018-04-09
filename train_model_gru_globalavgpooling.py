from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, concatenate, Input, BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import GRU, GlobalAveragePooling1D

from .metrics import keras_ndcg
from .set_variable import *

class MyModel:

    def __init__(self):
        self._create()

    def _create(self):
        EMBEDDING_DIMS = 50
        GRU_DIMS = 128

        DROPOUT_GRU = 0.0
        RNN_DROPOUT = 0.0
        print('Creating Model...')
        # EMBEDDING
        input_play = Input(shape=(SEQ_LEN,), dtype='int32', name='input_play')
        input_save = Input(shape=(SEQ_LEN,), dtype='int32', name='input_save')

        # Keras requires the total_dim to have 2 more dimension for "other" class
        embedding_layer = Embedding(
            input_dim=(EMBEDDING_CLASSES + 2), output_dim=EMBEDDING_DIMS,
            input_length=SEQ_LEN, mask_zero=False, trainable=True,
            name='emb')

        input_play_encoded = embedding_layer(input_play)
        input_save_encoded = embedding_layer(input_save)

        # biGru = Bidirectional(GRU(GRU_DIMS, return_sequences=False, activation='relu',  dropout=DROPOUT_GRU), name='gru1')
        gru1 = GRU(GRU_DIMS, return_sequences=True, dropout=DROPOUT_GRU, recurrent_dropout=RNN_DROPOUT, name='gru1')(input_play_encoded)
        gru2 = GRU(GRU_DIMS, return_sequences=True, dropout=DROPOUT_GRU, recurrent_dropout=RNN_DROPOUT, name='gru2')(input_save_encoded)

        play_encoded = GlobalAveragePooling1D(name='globplay')(gru1)
        save_encoded = GlobalAveragePooling1D(name='globsave')(gru2)

        # TIME_OF_DAY OHE
        ohe_tod = Input(shape=(TOTAL_TOD_BINS,), name='time_of_day_ohe')

        # DAY_OF_WEEK OHE
        ohe_dow = Input(shape=(TOTAL_DOW_BINS,), name='day_of_wk_ohe')

        # MERGE LAYERS
        print('Merging features...')
        merged = concatenate(
            [play_encoded, save_encoded, ohe_tod, ohe_dow], axis=1,
            name='concat')

        # FULLY CONNECTED LAYERS
        dense = Dense(1024, activation='relu', name='main_dense')(merged)
        pred = Dense(TARGET_CLASSES, activation='softmax', name='output')(dense)

        self.model = Model(
            inputs=[input_play, input_save, ohe_tod, ohe_dow],
            outputs=[pred])

        print(self.model.summary())

        return self.model

    def compile(self):

        self.model.compile(
                optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                metrics=[keras_ndcg(k=5)])
