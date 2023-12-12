import tensorflow as tf
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras import layers, losses
import os
from scipy import signal
from models.base_simsiame_Imran import *
import json
import io
# import plotly.graph_objects as go
# import pickle
# import random
from create_ds.nisqa_pair.temp import data_generator 

BATCH_SIZE = 64
EPOCHS=20


corpus_path = '/fs/scratch/PAS2622/Project_AI/Datasets/NISQA_Corpus/'

df_train_sim = pd.read_csv(corpus_path + 'NISQA_TRAIN_SIM/NISQA_TRAIN_SIM_file.csv')
df_train_live = pd.read_csv(corpus_path + 'NISQA_TRAIN_LIVE/NISQA_TRAIN_LIVE_file.csv')
df_train = pd.concat([df_train_sim, df_train_live], join='inner')

df_valid_sim = pd.read_csv(corpus_path + 'NISQA_VAL_SIM/NISQA_VAL_SIM_file.csv')
df_valid_live = pd.read_csv(corpus_path + 'NISQA_VAL_LIVE/NISQA_VAL_LIVE_file.csv')
df_vaid = pd.concat([df_valid_sim, df_valid_live], join='inner')

ds_train = tf.data.Dataset.from_generator(lambda: data_generator(dataframe=df_train, decimals=1), output_signature=
                (tf.TensorSpec(shape=(128, 1201, 2), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)))
ds_valid = tf.data.Dataset.from_generator(lambda: data_generator(dataframe=df_train, decimals=1), output_signature=
                (tf.TensorSpec(shape=(128, 1201, 2), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)))

ds_train = ds_train.batch(BATCH_SIZE).prefetch(1)
ds_valid = ds_valid.batch(BATCH_SIZE).prefetch(1)

# train_batch = list(next(iter(ds_train)))
# ds_instance = iter(ds_train)
# ftr_mat, mos_labl = next(ds_instance)
# ftr_mat, mos_labl = next(ds_instance)
# print(mos_labl.numpy())
# exit()

model = SimSiameseBasedModel()

# x_audio_0_ds = ds_train.map(lambda x: x['ref_f_abs'])
# x_audio_1_ds = ds_train.map(lambda x: x['deg_f_abs'])
# x_mos_ds = ds_train.map(lambda x: x['mos'])

# model.normalizer_audio_0.adapt(x_audio_0_ds)
# model.normalizer_audio_1.adapt(x_audio_1_ds)
# model.normalizer_mos.adapt(x_mos_ds)

opt_ctr = tf.keras.optimizers.Adam(learning_rate=0.001)
opt_cls = tf.keras.optimizers.Adam(learning_rate=0.001)
opt = tf.keras.optimizers.Adamax(learning_rate=0.001)
model.compile(optimizer=opt, loss=losses.SparseCategoricalCrossentropy(), opt_ctr=opt_ctr, opt_cls=opt_cls)

history = model.fit(ds_train,
                    steps_per_epoch = 86,
                    epochs=EPOCHS,
                    validation_data=(ds_valid),)
