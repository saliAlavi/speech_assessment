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

tf.random.set_seed(1)
np.random.seed(1)

# For distributed training purposes
print(tf.config.list_physical_devices('GPU'))
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BATCH_SIZE_PER_REPLCIA = 16
GLOBAL_BATCH_SIZE  = BATCH_SIZE_PER_REPLCIA * strategy.num_replicas_in_sync

# Gather NISQA voice dataset
dataset_name = f'nisqa/default'

train_sim, test_sim = tfds.load(
    dataset_name,
    split=['train_sim', 'test_sim']
)

# Spectrogram Visualization
plt.figure(figsize=(10, 4))
for i, audio  in enumerate(train_sim.take(3)):
    ax = plt.subplot(3, 1, i + 1)
    plt.imshow(audio['log_mel_spectogram'], origin="lower", cmap=plt.get_cmap("magma"))
    plt.title('mos = ' + str(audio['mos'].numpy()))
    plt.axis("off") 

# Audio Augmentation
spect_dataset = train_sim.map(lambda x: x['log_mel_spectogram'])
norm_layer = tf.keras.layers.Normalization(axis=None)
norm_layer.adapt(spect_dataset)

def audio_augment(audio):
    # audio_n = norm_layer(audio)
    audio_n = audio
    noise_std_audio = 0.1

    audio_0_n = audio_n + tf.random.normal(tf.shape(audio_n), mean=0, stddev=noise_std_audio)
    audio_1_n = audio_n + tf.random.normal(tf.shape(audio_n), mean=0, stddev=noise_std_audio)
    
    audio_pair = tf.concat([tf.expand_dims(audio_0_n, axis=2), tf.expand_dims(audio_1_n, axis=2)], axis=2)
    return audio_pair

# Dataset of Pairs
AUTO = tf.data.experimental.AUTOTUNE

train_dataset = (
    spect_dataset
    .shuffle(1024, seed=0)
    .map(audio_augment, num_parallel_calls=AUTO)
    .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    # .prefetch(AUTO)
)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

# Visualization 
sample_pairs = next(iter(train_dataset))

plt.figure(figsize=(10, 3))
for i in range(3):
    audio_pair = sample_pairs[i]
    plt.subplot(3,2,(2*i)+1)
    plt.imshow(audio_pair[:,:,0], origin="lower", cmap=plt.get_cmap("magma"))
    plt.axis('off')
    plt.subplot(3,2,(2*i)+2)
    plt.imshow(audio_pair[:,:,1], origin="lower", cmap=plt.get_cmap("magma"))
    plt.axis('off')
    plt.savefig('/fs/scratch/PAS2622/ssl_based/new_stream/plain_aug_audio_pairs.png')
    
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

# Predictor (h)
def get_predictor():
    inputs = tf.keras.layers.Input((2048, ))
    
    x = tf.keras.layers.Dense(512, use_bias=False)(inputs) # layer 1
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    p = tf.keras.layers.Dense(2048)(x) # layer 2

    h = tf.keras.Model(inputs, p)

    return h

# Model Summary
get_encoder().summary()
get_predictor().summary()

with strategy.scope():
    # Define the Cosine Loss function
    def loss_func(p, z):
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)
        return tf.reduce_sum((p*z), axis=1)

    def compute_loss(p, z):
        per_example_loss = loss_func(p, z)
        avg_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        return avg_loss

with strategy.scope():
    # A model, an optimizer, and a checkpoint must be created under `strategy.scope`.
    f = get_encoder()
    h = get_predictor()
    
    decay_steps = 500
    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.01, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6)

# Training Step
def train_step(ds, f, h, optimizer):
    audio_1 = ds[:,:,:,0]
    audio_2 = ds[:,:,:,1]
    
    with tf.GradientTape() as tape:
        z1, z2 = f(audio_1), f(audio_2)
        p1, p2 = h(z1), h(z2)
        # loss = loss_func(p1, z2)/2 + loss_func(p2, z1)/2
        loss = compute_loss(p1, z2)/2 + compute_loss(p2, z1)/2
    
    learnable_params = f.trainable_variables + h.trainable_variables
    gradients = tape.gradient(loss, learnable_params)
    optimizer.apply_gradients(zip(gradients, learnable_params))

    return loss

@tf.function
def distributed_train_step(dataset_inputs, f, h, optimizer):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs, f, h, optimizer))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)
  
def train_simsiam(f, h, dataset, optimizer, epochs=100):
    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in tqdm(range(epochs)):
        for batch in dataset:
            loss = distributed_train_step(batch, f, h, optimizer)
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))

        if epoch % 1 == 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

    return epoch_wise_loss, f, h

epoch_wise_loss, f, h  = train_simsiam(f, h, train_dist_dataset, optimizer, epochs=50)

# Model Performance Analysis
plt.figure()
plt.plot(epoch_wise_loss)
plt.grid()
plt.savefig('/fs/scratch/PAS2622/ssl_based/new_stream/plain_aug_curve.png')

f.save_weights("/fs/scratch/PAS2622/ssl_based/new_stream/plain_projection.h5")
h.save_weights("/fs/scratch/PAS2622/ssl_based/new_stream/plain_prediction.h5")