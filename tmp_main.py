import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow.keras import layers, losses
import os
from scipy import signal
from models import *
import json
import pandas as pd
import io
import plotly.graph_objects as go
import pickle
import random

print('Start')
BATCH_SIZE = 4
EPOCHS=500
dataset_name = f'nisqa/default'
# (ds_train, ds_test), dataset_info = tfds.load(dataset_name, split=['train','test'], shuffle_files=True,with_info=True,)
(ds_train, ds_test), dataset_info = tfds.load(dataset_name, split=['train_sim','test_sim'], shuffle_files=True,with_info=True,)
ds_train= ds_train.prefetch(tf.data.experimental.AUTOTUNE).shuffle(BATCH_SIZE).batch(BATCH_SIZE, drop_remainder=True)
ds_test=  ds_test.prefetch(tf.data.experimental.AUTOTUNE).shuffle(BATCH_SIZE).batch(BATCH_SIZE, drop_remainder=True)

model = SimSiameseBasedModel()

x_audio_0_ds = ds_train.map(lambda x: x['ref_f_abs'])
x_audio_1_ds = ds_train.map(lambda x: x['deg_f_abs'])
x_mos_ds = ds_train.map(lambda x: x['mos'])

model.normalizer_audio_0.adapt(x_audio_0_ds)
model.normalizer_audio_1.adapt(x_audio_1_ds)
model.normalizer_mos.adapt(x_mos_ds)

learning_rate=1e-3
opt_ctr = tf.keras.optimizers.Adam(learning_rate=learning_rate)
opt_cls = tf.keras.optimizers.Adam(learning_rate=learning_rate)
opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
model.compile(optimizer=opt, loss=losses.SparseCategoricalCrossentropy(), opt_ctr=opt_ctr, opt_cls=opt_cls)

history = model.fit(ds_train,
                    epochs=EPOCHS,
                    shuffle=True,
                    validation_data=(ds_test),)
