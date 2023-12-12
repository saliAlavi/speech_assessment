import tensorflow as tf
print(tf.__version__)

import tensorflow_datasets as tfds
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os

from tqdm import tqdm

tfds.disable_progress_bar()
print(tf.config.list_physical_devices('GPU'))

# Gather NISQA voice dataset
dataset_name = f'nisqa/default'

train_sim, train_live, test_sim, test_live = tfds.load(
    dataset_name,
    split=['train_sim', 'train_live', 'test_sim', 'test_live']
)

# Data preparation
spect_dataset = train_sim.map(lambda x: x['log_mel_spectogram'])
norm_layer = tf.keras.layers.Normalization(axis=None)
norm_layer.adapt(spect_dataset)

def my_tf_round(x, decimals=1):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

def mapping_func(x):
    ftr_mat = x['log_mel_spectogram']
    ftr_mat_norm = norm_layer(ftr_mat)

    mos = x['mos']
    truncated_mos = my_tf_round(mos, 1)
    categorical_labl = int((truncated_mos - 1) * 10) # 1 is the lower mos offset
    
    return (ftr_mat_norm, categorical_labl)

# Dataset of Pairs
BATCH_SIZE = 16
AUTO = tf.data.experimental.AUTOTUNE

training_ds = (
    train_sim
    .map(mapping_func, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

validation_ds = (
    test_sim
    .map(mapping_func, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Encoder (f)
def get_encoder():
    base_model = tf.keras.applications.ResNet50(include_top=False,
        weights=None, input_shape=(128, 1201, 1))
    base_model.trainable = True

    inputs = tf.keras.layers.Input((128, 1201, 1))
    x = base_model(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2048, activation='relu', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    z = tf.keras.layers.Dense(2048)(x)

    f = tf.keras.Model(inputs, z)

    return f

get_encoder().summary()


# We now load up the pre-trained weights
projection = get_encoder()
projection.load_weights('/fs/scratch/PAS2622/ssl_based/new_stream/plain_projection.h5')

# Create a sub-model for extracting features
rn50 = tf.keras.Model(projection.input, projection.layers[2].output)
rn50.summary()

# # per-channel std
# output = projection.predict(temp_ds)
# output_std = tf.math.reduce_std(output, axis=0)
# print(tf.math.reduce_mean(output_std))

# Classifier 
def get_linear_classifier(feature_backbone, trainable=False):
    inputs = tf.keras.layers.Input(shape=(128, 1201, 1))
    
    feature_backbone.trainable = trainable
    x = feature_backbone(inputs, training=False)
    outputs = tf.keras.layers.Dense(41, activation="softmax", )(x) # total 41 classes [1.0, 1.1, ..., 4.9, 5.0]
    linear_model = tf.keras.Model(inputs, outputs)

    return linear_model

get_linear_classifier(rn50).summary()


def plot_progress(hist):
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="validation_loss")
    plt.plot(hist.history["accuracy"], label="training_accuracy")
    plt.plot(hist.history["val_accuracy"], label="validation_accuracy")
    plt.title("Training Progress")
    plt.ylabel("accuracy/loss")
    plt.xlabel("epoch")
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig('mos_training.png')
    
# Early Stopping to prevent overfitting
early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
                                                 patience=5, verbose=2, 
                                                 restore_best_weights=True)

# Get linear model and compile
tf.keras.backend.clear_session()
model = get_linear_classifier(rn50)
model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                     optimizer="adam")

# Train 
history = model.fit(training_ds,
                 validation_data=validation_ds,
                 epochs=100)
                #  callbacks=[early_stopper])

plot_progress(history)

_, acc = model.evaluate(validation_ds)
print('Validation accuracy:', round(acc*100, 2))