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

print(tf.config.list_physical_devices('GPU'))
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BATCH_SIZE_PER_REPLCIA = 16
GLOBAL_BATCH_SIZE  = BATCH_SIZE_PER_REPLCIA * strategy.num_replicas_in_sync

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

def mapping_func(x):
    ftr_mat = x['log_mel_spectogram']
    ftr_mat_norm = norm_layer(ftr_mat)
    mos_labl = x['mos']
    return (ftr_mat_norm, mos_labl)

# Dataset of Pairs
AUTO = tf.data.experimental.AUTOTUNE

training_ds = (
    train_sim
    .shuffle(1024, seed=0)
    .map(mapping_func, num_parallel_calls=AUTO)
    .batch(GLOBAL_BATCH_SIZE)
    # .prefetch(AUTO)
)

validation_ds = (
    test_sim
    .map(mapping_func, num_parallel_calls=AUTO)
    .batch(GLOBAL_BATCH_SIZE)
    # .prefetch(AUTO)
)

# Encoder (f)
def get_encoder():
    base_model = tf.keras.applications.ResNet50(include_top=False,
        weights=None, input_shape=(128, 1201, 1))
    base_model.trainable = True

    inputs = tf.keras.layers.Input((128, 1201, 1))
    x = base_model(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x) 
    
    x = tf.keras.layers.Dense(2048, use_bias=False)(x) # layer 1
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(2048, use_bias=False)(x) # layer 2
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    z = tf.keras.layers.Dense(2048)(x) # layer 3
    # z = tf.keras.layers.BatchNormalization()(x)

    f = tf.keras.Model(inputs, z)
    return f

get_encoder().summary()

with strategy.scope():
    # We now load up the pre-trained weights
    projection = get_encoder()
    projection.load_weights('plain_projection.h5')

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
    x = tf.keras.layers.Dense(2048)(x)
    x = tf.keras.layers.Dense(2048)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    linear_model = tf.keras.Model(inputs, outputs)
    return linear_model

get_linear_classifier(rn50).summary()


def plot_progress(hist):
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="validation_loss")

    plt.title("Training Progress")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig('plain_aug_regression.png')
    
# Early Stopping to prevent overfitting
early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
                                                 patience=5, verbose=2, 
                                                 restore_best_weights=True)

# Get linear model and compile
tf.keras.backend.clear_session()
with strategy.scope():
    model = get_linear_classifier(rn50)
    model.compile(loss="mean_squared_error", optimizer="adam")

# Train 
history = model.fit(training_ds,
                 validation_data=validation_ds,
                 epochs=25)
                #  callbacks=[early_stopper])
plot_progress(history)

model.save_weights("/fs/scratch/PAS2622/ssl_based/new_stream/plain_regression.h5")

# Check predicted values
predictions = model.predict(validation_ds)
print('Mean of predicted mos values = ', np.mean(predictions))
print('Variance in predicted mos values = ', np.power(np.std(predictions),2))
print('Maximum of predicted mos values = ', np.max(predictions))
print('Minimum of predicted mos values = ', np.min(predictions))
print('Range of predicted mos values = ', np.max(predictions) - np.min(predictions))