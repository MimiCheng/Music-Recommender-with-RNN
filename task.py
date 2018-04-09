import sys
import tensorflow as tf
import argparse
import ast

from datetime import datetime
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.lib.io import file_io

from train_model import MyModel
from utils import BaseCSVDataGenerator, GcsModelCheckpoint
from set_variable import *

class DataGenerator(BaseCSVDataGenerator):

    def __init__(self, directory, batch_size=1024, chunk_size=20000):
        super().__init__(directory, batch_size, chunk_size)
        self.EMB_FEATURES = [
                'play_song', 'save', 'fav_song', 'addtoplaylist', 'inplaylist',
                'lyrics']

    def _to_id(self, ids):
        ids = ast.literal_eval(ids)
        if ids == []:
            return [len(emb2idx) - 1]
        else:
            return [emb2idx[str(item_id)] for item_id in ids]

    def preprocess(self, df):
        time_of_day = df["time_of_day"].map(tod_dict).values
        day_of_week = df["day_of_week"].map(dow_dict).values

        label = df["target"].map(target2idx).values
        play_song = df['play_song'].map(self._to_id).values
        save = df['save'].map(self._to_id).values

        # one hot for categorical features
        ohe_label = to_categorical(label, num_classes=TARGET_CLASSES)
        ohe_tod = to_categorical(time_of_day, num_classes=TOTAL_TOD_BINS)
        ohe_dow = to_categorical(day_of_week, num_classes=TOTAL_DOW_BINS)

        # padding
        play_song_pad = pad_sequences(
                play_song, maxlen=SEQ_LEN, padding='pre', truncating='pre')
        save_pad = pad_sequences(
                save, maxlen=SEQ_LEN, padding='pre', truncating='pre')

        X = [play_song_pad, save_pad, ohe_tod, ohe_dow]
        y = ohe_label

        return X, y

def main(unused_argv):

    TOT_WORKERS = 2

    dnn = MyModel()
    dnn.compile()

    train_gen = DataGenerator(FLAGS.train_dir, FLAGS.batch_size)
    test_gen = DataGenerator(FLAGS.dev_dir, FLAGS.batch_size)

    cols = ['seq', 'event_dt', 'ssoid', 'os_name', 'device_cate', 'country',
            'region_name',  'time_of_day', 'day_of_week', 'target', 'play_song',
            'play_artist', 'save', 'fav_song', 'fav_album', 'fav_artist',
            'fav_playlist', 'addtoplaylist', 'removetoplaylist', 'inplaylist',
            'lyrics']
    kwargs = dict(error_bad_lines=False, header=0, names=cols)

    dt = datetime.now().strftime('%Y%m%d')

    checkpoint_dir = FLAGS.job_dir + 't/checkpoints'
    tf.gfile.MakeDirs(checkpoint_dir)

    final_weight = str(FLAGS.model_dir) + FLAGS.weight_name

    if FLAGS.weight_name != 'False':
        dnn.model.load_weights(str(FLAGS.weight_name))
        print('load weight {} successfully'.format(final_weight))
    else:
        print('weight is not loaded')

    # callbacks
    tb = TensorBoard(log_dir=str(FLAGS.job_dir) + '/logs', batch_size=FLAGS.batch_size)
    checkpoint = GcsModelCheckpoint(
            checkpoint_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss', verbose=0, save_best_only=False,
            save_weights_only=True, mode='min', period=1)

    reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    callbacks = [checkpoint, reduce_lr, tb]

    model_json = dnn.model.to_json()
    with file_io.FileIO(str(FLAGS.job_dir) + "/model{}.json".format(dt), "w") as json_file:
        json_file.write(model_json)
    #fit model
    dnn.model.fit_generator(
            train_gen.generate(**kwargs),
            steps_per_epoch=train_gen.steps_per_epoch(**kwargs),
            validation_data=test_gen.generate(**kwargs),
            validation_steps=test_gen.steps_per_epoch(**kwargs),
            callbacks=callbacks,
            workers=TOT_WORKERS,
            epochs=FLAGS.train_epochs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir', type=str, help='Base directory for the model.')
    parser.add_argument(
        '--job_dir', type=str, help='Base directory for the job.')
    parser.add_argument(
        '--train_epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='Number of examples per batch.')
    parser.add_argument(
        '--train_dir', type=str, help='Path to the training data.')
    parser.add_argument(
        '--dev_dir', type=str, help='Path to the validation data.')
    parser.add_argument(
        '--weight_name', type=str, help='Set False if no weight file to load otherwise input weight name to load')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
