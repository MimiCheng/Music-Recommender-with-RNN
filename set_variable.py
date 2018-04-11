"""
Created on Tue Feb 20 2018
@author: Patcharin Cheng
"""
import tensorflow as tf
import pandas as pd

import os

job_dir = FLAG.job_dir

def get_metadata(path, key, val):
    csv_files = tf.gfile.Glob(path)
    df = pd.read_csv(tf.gfile.Open(csv_files[0]))
    if key == "real_world_id":
        return {str(row[key]): row[val] for _, row in df.iterrows()}
    else:
        return {row[key]: row[val] for _, row in df.iterrows()}

target2idx = get_metadata("{0}/datasets/metadata/targets/*.csv".format(job_dir), "real_world_id", "model_id")
emb2idx = get_metadata("{0}/datasets/metadata/emb_songs/*.csv".format(job_dir), "real_world_id", "model_id")
idx2target = get_metadata("{0}/datasets/metadata/targets/*.csv".format(job_dir), "model_id", "real_world_id")
idx2emb = get_metadata("{0}/datasets/metadata/emb_songs/*.csv".format(job_dir), "model_id", "real_world_id")

print('number of unique song is {}'.format(len(emb2idx)))
print('number of target class is {}'.format(len(target2idx)))

# create DOW mapping
dow = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_id = [0, 1, 2, 3, 4, 5, 6]
dow_dict = dict(zip(dow, dow_id))

# create TOD mapping
tod = ["Midnight", "Late Night", "Early Morning", "Morning", "Noon", "Afternoon", "Evening", "Night"]
tod_id = [0, 1, 2, 3, 4, 5, 6, 7]
tod_dict = dict(zip(tod, tod_id))

SEQ_LEN = os.getenv("SEQ_LEN", 20)
EMBEDDING_CLASSES = os.getenv("EMBEDDING_CLASSES", len(emb2idx))
TARGET_CLASSES = os.getenv("TARGET_CLASSES", len(target2idx))
TOTAL_TOD_BINS = os.getenv("TOTAL_TOD_BINS", 8)
TOTAL_DOW_BINS = os.getenv("TOTAL_DOW_BINS", 7)
