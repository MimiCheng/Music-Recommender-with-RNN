from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, concatenate, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import GRU, Conv1D, MaxPooling1D, BatchNormalization

from .metrics import keras_ndcg
from .set_variable import *

class MyModel:

    def __init__(self):
        self._create()

    def _create(self):
        EMBEDDING_DIMS = 50
        GRU_DIMS = 64
        DROPOUT_FC = 0.2
        DROPOUT_GRU = 0.2
        DROPOUT_EMB = 0.2
        # Convolution
        kernel_size = 3
        filters = 32
        pool_size = 2

        print('Creating Model...')
        # EMBEDDING
        input_play = Input(shape=(SEQ_LEN,), dtype='int32', name='input_play')

        # Keras requires the total_dim to have 2 more dimension for "other" class
        embedding_layer = Embedding(
            input_dim=(EMBEDDING_CLASSES + 2), output_dim=EMBEDDING_DIMS,
            input_length=SEQ_LEN, mask_zero=False, trainable=True,
            name='emb')(input_play)
        drop_emb = Dropout(DROPOUT_EMB, name='dropout_emb')(embedding_layer)

        conv = Conv1D(filters, kernel_size,
                      padding='same',
                      activation='relu',
                      strides=1, name='conv1')(drop_emb)

        maxpool = MaxPooling1D(pool_size=pool_size, name='maxpool1')(conv)

        gru = GRU(
            GRU_DIMS, dropout=DROPOUT_GRU,
            name='gru1')(maxpool)

        # TIME_OF_DAY OHE
        ohe1 = Input(shape=(TOTAL_TOD_BINS,), name='time_of_day_ohe')

        # DAY_OF_WEEK OHE
        ohe2 = Input(shape=(TOTAL_DOW_BINS,), name='day_of_wk_ohe')

        # MERGE LAYERS
        print('Merging features...')

        merged = concatenate(
            [gru, ohe1, ohe2], axis=1,
            name='concat')

        # FULLY CONNECTED LAYERS
        dense = Dense(128, activation='relu', name='main_dense')(merged)
        bn = BatchNormalization(name='bn_fc1')(dense)
        drop = Dropout(DROPOUT_FC, name='dropout1')(bn)
        dense = Dense(64, activation='relu', name='dense2')(drop)
        drop = Dropout(DROPOUT_FC, name='dropout2')(dense)
        dense = Dense(32, activation='relu', name='dense3')(drop)
        pred = Dense(TARGET_CLASSES, activation='softmax', name='output')(dense)

        self.model = Model(
            inputs=[input_play, ohe1, ohe2],
            outputs=[pred])
        print(self.model.summary())

        return self.model

    def compile(self):
        self.model.compile(
                optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                metrics=[keras_ndcg(k=5)])
