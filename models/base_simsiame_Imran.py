# is stop gradient implemented at the right place?
# why are we concatenating representations from 2 audios when feeding to classifier?
# how do we know when an epoch has ended
# look at simsiamese code
# look at variance per channel


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow import keras
import copy

BATCH_SIZE = 64

class EncoderBase(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_0 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='conv_0', input_shape=(128,1201,1))
        self.maxpool_0 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.conv_1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='conv_1')
        self.maxpool_1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.conv_2 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='conv_2')
        self.maxpool_2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.conv_3 = layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_3')
        self.maxpool_3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.flatten = layers.Flatten()

        self.dense_0 = layers.Dense(units=128, activation='relu')

    def call(self, x, training=False):
        x = self.conv_0(x)
        x = self.maxpool_0(x)

        x = self.conv_1(x)
        x = self.maxpool_1(x)

        x = self.conv_2(x)
        x = self.maxpool_2(x)

        x = self.conv_3(x)
        x = self.maxpool_3(x)

        x=self.flatten(x)
        x = self.dense_0(x)

        return x


class ProbeHead(layers.Layer):
    def __init__(self, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dense_0 = layers.Dense(units=128, activation='relu')
        self.do_0 = layers.Dropout(rate=dropout_rate)

        self.dense_1 =layers.Dense(units=64, activation='relu')
        self.do_1 =layers.Dropout(rate=dropout_rate)

        self.dense_2 =layers.Dense(units=128, activation='relu')

    def call(self, x, training=False):
        x = self.dense_0(x)
        x = self.do_0(x, training=training)
        
        x = self.dense_1(x)
        x = self.do_1(x,training=training)

        x = self.dense_2(x)

        return x


class ClassifierHead(layers.Layer):
    def __init__(self, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dense_0 = layers.Dense(units=128, activation='relu')
        self.do_0 = layers.Dropout(rate=dropout_rate)

        self.dense_1 =layers.Dense(units=128, activation='relu')
        self.do_1 =layers.Dropout(rate=dropout_rate)

        self.dense_2 =layers.Dense(units=1)

    def call(self, x, training=False):
        x = self.dense_0(x)
        x = self.do_0(x, training=training)

        x = self.dense_1(x)
        x = self.do_1(x,training=training)

        x = self.dense_2(x)

        return x


class SimSiameseBasedModel(keras.Model):
    def __init__(self):
        super(SimSiameseBasedModel, self).__init__()

        self.train_contrastive = True

        self.encoder_base = EncoderBase()
        self.probe = ProbeHead()
        self.classifier_head = ClassifierHead(dropout_rate=0.05)

        ## Comment: normalization adversely effects speech loudness (ref: DNSMOS paper)
        # self.normalizer_audio_0 = tf.keras.layers.Normalization(axis=[1,2])
        # self.normalizer_audio_1 = tf.keras.layers.Normalization(axis=[1,2])
        # self.normalizer_mos = tf.keras.layers.Normalization(axis=1)

        self.classifier_loss = keras.losses.MeanSquaredError()

        self.lambda_ = .9

        # Contrastive Parameters
        self.temperature = 0.1

    def compile(self, opt_ctr=keras.optimizers.Adam(learning_rate=1e-3), opt_cls=keras.optimizers.Adam(learning_rate=1e-3), **kwargs):
        super().compile(**kwargs)

        self.opt_contrastive = opt_ctr
        self.opt_classifier = opt_cls

        self.loss_classifier_metric = keras.metrics.Mean(name="loss")
        self.loss_contrastive = keras.metrics.Mean(name="loss_ctr")

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        
        x,mos = data
        audio_0 = x[:,:,:,0]
        audio_1 = x[:,:,:,1]

        audio_0 = tf.expand_dims(audio_0, axis=3)
        audio_1 = tf.expand_dims(audio_1, axis=3)
        
        # mos = tf.reshape(data['mos'], [-1])
        # audio_0 = data['ref_f_abs']
        # audio_1 = data['deg_f_abs']
        # audios = tf.stack([audio_0, audio_1], axis=3)

        # audio_0 = self.normalizer_audio_0(audio_0)
        # audio_1 = self.normalizer_audio_0(audio_1)
        # mos = self.normalizer_mos(mos)
        # if self.train_contrastive:
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            # Compute the loss value
            # (the loss function is configured in `compile()`

            # noise_std_audio = 0.01
            # audio_0_n = audio_0 + tf.random.normal(tf.shape(audio_0), mean=0, stddev=noise_std_audio)
            # audio_0_n = audio_1 + tf.random.normal(tf.shape(audio_1), mean=0, stddev=noise_std_audio)
            # audio_1_n = audio_1 + tf.random.normal(tf.shape(audio_1), mean=0, stddev=noise_std_audio)

            # audio_0_n = tf.expand_dims(audio_0_n, axis=3)
            # audio_1_n = tf.expand_dims(audio_1_n, axis=3)

            p_0, z_0 = self(audio_0, contrastive=True, training=True) 
            p_1, z_1 = self(audio_1, contrastive=True, training=True) 

            # noise_std_mos = 0.05
            # mos_n = mos + tf.random.normal(tf.shape(mos), mean=0, stddev=noise_std_mos)
            # mos_n = mos

            # loss_0 = self.SimSiameseLoss(p_0, z_1, mos_n, mos_n, 0.1, 1)
            # loss_1 = self.SimSiameseLoss(p_1, z_0, mos_n, mos_n, 0.1, 1)
            loss_0 = self.D(p_0, z_1)
            loss_1 = self.D(p_1, z_0)
            loss_cont = (loss_0+loss_1)/2 

        # Compute gradients
        gradients = tape.gradient(
            loss_cont,
            self.encoder_base.trainable_weights + self.probe.trainable_weights
        )

        self.opt_contrastive.apply_gradients(
            zip(
                gradients,
                self.encoder_base.trainable_weights + self.probe.trainable_weights
            )
        )

        # classifier learning
        # if not self.train_contrastive:


        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            # Compute the loss value
            # (the loss function is configured in `compile()`

            # audio_0 = tf.expand_dims(audio_0, axis=3)
            # audio_1 = tf.expand_dims(audio_1, axis=3)

            # pred = self([audio_0, audio_1], contrastive = False, training=True) 
            pred = self(audio_0, contrastive = False, training=True) 

            loss = self.classifier_loss(mos, pred)

        # Compute gradients
        gradients = tape.gradient(
            loss,
            self.classifier_head.trainable_weights#+self.encoder_base.trainable_weights
        )

        self.opt_classifier.apply_gradients(
            zip(
                gradients,
                self.classifier_head.trainable_weights#+self.encoder_base.trainable_weights
            )
        )
        # Update weights
        # Update metrics (includes the metric that tracks the loss)
        # Return a dict mapping metric names to current value
        self.loss_classifier_metric.update_state(loss)
        self.loss_contrastive.update_state(loss_cont)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # mos = tf.reshape(data['mos'], [-1])
        # audio_0 = data['ref_f_abs']
        # audio_1 = data['deg_f_abs']
        # audios = tf.stack([audio_0, audio_1], axis=3)

        # audio_0 = self.normalizer_audio_0(audio_0)
        # audio_1 = self.normalizer_audio_0(audio_1)
        # mos = self.normalizer_mos(mos)

        # audio_0 = tf.expand_dims(audio_0, axis=3)
        # audio_1 = tf.expand_dims(audio_1, axis=3)

        x,mos = data
        audio_0 = x[:,:,0]
        audio_1 = x[:,:,1]

        audio_0 = tf.expand_dims(audio_0, axis=3)
        audio_1 = tf.expand_dims(audio_1, axis=3)
        
        pred = self([audio_0, audio_1], contrastive=False, training=False) # WHY ARE WE CONCATENATING 2 AUDIOS
        loss = self.classifier_loss(mos, pred)

        self.loss_classifier_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def call(self, data_inp, training=False, contrastive=False):
        if contrastive:
            audio = data_inp
            p = self.encoder_base(audio)
            z = self.probe(p, training=training)
            return p, z
        else:
            # audio_0, audio_1 = data_inp
            audio_0 = data_inp

            x_0 = self.encoder_base(audio_0)
            # x_1 = self.encoder_base(audio_1)
            # x = tf.concat([x_0, x_1], axis=1)
            x = x_0
            
            x = self.classifier_head(x, training=training)
            return x

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_classifier_metric,
            self.loss_contrastive,
        ]


    @staticmethod
    def SimSiameseLoss(p, z, m_p, m_z, threshold, lambda_):
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)

        m_p = tf.expand_dims(m_p, axis=1)
        m_z = tf.expand_dims(m_z, axis=1)
        m_diff = m_p - tf.transpose(m_z)

        threshold = tf.constant([threshold])
        cond_p = tf.cast(tf.greater_equal(tf.math.abs(m_diff), threshold),tf.float32)
        cond_n = tf.cast(tf.less(tf.math.abs(m_diff), threshold), tf.float32)

        projection = tf.linalg.matmul(p, z, transpose_b=True)

        loss_p = cond_p*tf.math.abs(m_diff)*projection
        loss_n = cond_n*m_diff*projection
        loss_p = tf.math.exp(-loss_p/tf.reduce_sum(projection, keepdims=True))
        loss_n = tf.math.exp(-loss_n / tf.reduce_sum(projection, keepdims=True))
        loss_p = tf.math.log(tf.reduce_sum(loss_p))
        loss_n = tf.math.log(tf.reduce_sum(loss_n))
        loss = loss_p - lambda_*loss_n
        # loss = tf.reduce_sum(loss,axis=1)
        loss = tf.reduce_mean(loss)

        return loss

    @staticmethod
    def D(p, z):
        
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)

        # print('p shape after normalization: ', np.shape(p))
        # print('z shape after normalization: ', np.shape(z))
        # exit()
        
        # m_p = tf.expand_dims(m_p, axis=1)
        # m_z = tf.expand_dims(m_z, axis=1)
        # m_diff = m_p - tf.transpose(m_z)

        # threshold = tf.constant([threshold])
        # cond_p = tf.cast(tf.greater_equal(tf.math.abs(m_diff), threshold),tf.float32)
        # cond_n = tf.cast(tf.less(tf.math.abs(m_diff), threshold), tf.float32)

        projection = tf.linalg.matmul(p, z, transpose_b=True)

        # print('projection shape: ', np.shape(projection))
        # exit()
        
        # loss_p = cond_p*tf.math.abs(m_diff)*projection
        # loss_n = cond_n*m_diff*projection
        # loss_p = tf.math.exp(-loss_p/tf.reduce_sum(projection, keepdims=True))
        # loss_n = tf.math.exp(-loss_n / tf.reduce_sum(projection, keepdims=True))
        # loss_p = tf.math.log(tf.reduce_sum(loss_p))
        # loss_n = tf.math.log(tf.reduce_sum(loss_n))
        # loss = loss_p - lambda_*loss_n
        loss = projection
        loss = tf.math.reduce_sum(loss,axis=1)
        loss = tf.reduce_mean(loss)
        loss = -1*loss
        
        print('loss: ', loss)

        return loss