import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow import keras
import copy
import tensorflow_hub as hub

BATCH_SIZE = 4

PROBE_DIM = 50
PROBE_LATENT_DIM=PROBE_DIM//2

#wd from https://keras.io/examples/vision/simsiam/
WEIGHT_DECAY = 0.0005


class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides, nonlinearity='relu',**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.kernel_size=kernel_size
        self.strides =strides
        self.nonlinearity=nonlinearity

        self.l0 = layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(strides, strides), padding='same', activation=None)
        self.l1 = layers.BatchNormalization(axis=3)
        self.l2 = layers.Activation(nonlinearity)

    def call(self, x, training=False):

        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)

        return x


class ResBlock(layers.Layer):
    def __init__(self, filters, match_filter_size = False, nonlinearity = 'relu',**kwargs):
        super().__init__(**kwargs)
        self.match_filter_size = match_filter_size
        self.filters = filters
        self.nonlinearity = nonlinearity
        if match_filter_size:
            self.l1 = ConvBlock(filters,1, 2, nonlinearity)
        else:
            self.l1 = ConvBlock(filters, 1, 1, nonlinearity)

        self.l2 = ConvBlock(filters, 3, 1, nonlinearity)
        self.l3 = ConvBlock(filters*4, 1, 1, nonlinearity)

        if self.match_filter_size:
            self.l_skip = layers.Conv2D(self.filters*4, 1, strides = (2,2), activation=None, padding='same')
        else:
            self.l_skip = layers.Conv2D(self.filters * 4, 1, strides=(1, 1), activation=None, padding='same')

        self.bn = layers.BatchNormalization(axis=3)
        self.add = layers.Add()
        self.activation = layers.Activation(self.nonlinearity)

    def call(self, x, training=False):
        x_skip = x

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        x_skip = self.l_skip(x_skip)

        x_skip = self.bn(x_skip)
        x = self.add([x, x_skip])
        x=self.activation(x)

        return x


class ResNet50(layers.Layer):
    def __init__(self, out_shape= 2048, nonlinearity='relu', **kwargs):
        super().__init__(**kwargs)
        self.nonlinearity = nonlinearity
        self.out_shape = out_shape

        filter_size = 64
        self.s1_0 = layers.Conv2D(filter_size, kernel_size = (7, 7), strides = (2,2),padding='same', activation=None)
        self.s1_1 = layers.BatchNormalization(axis=3)
        self.s1_2 = layers.Activation(self.nonlinearity)
        self.s1_3 = layers.MaxPooling2D((3, 3), strides=(2, 2))

        filter_size *= 2
        self.s2_0 = ResBlock(filter_size, False, nonlinearity)
        self.s2_1 = ResBlock(filter_size, False, nonlinearity)
        self.s2_2 = ResBlock(filter_size, False, nonlinearity)

        filter_size *= 2
        self.s3_0 = ResBlock(filter_size, True, nonlinearity)
        self.s3_1 = ResBlock(filter_size, False, nonlinearity)
        self.s3_2 = ResBlock(filter_size, False, nonlinearity)
        self.s3_3 = ResBlock(filter_size, False, nonlinearity)

        filter_size *= 2
        self.s4_0 = ResBlock(filter_size, True, nonlinearity)
        self.s4_1 = ResBlock(filter_size, False, nonlinearity)
        self.s4_2 = ResBlock(filter_size, False, nonlinearity)
        self.s4_3 = ResBlock(filter_size, False, nonlinearity)
        self.s4_4= ResBlock(filter_size, False, nonlinearity)
        self.s4_5 = ResBlock(filter_size, False, nonlinearity)

        filter_size *= 2
        self.s5_0 = ResBlock(filter_size, True, nonlinearity)
        self.s5_1 = ResBlock(filter_size, False, nonlinearity)
        self.s5_2 = ResBlock(filter_size, False, nonlinearity)
        self.s5_3 = ResBlock(filter_size, False, nonlinearity)

        self.sf_0 = layers.GlobalAveragePooling2D()
        self.sf_1 = layers.Flatten()
        self.sf_2 = layers.Dense(units=out_shape, activation=nonlinearity)


    def call(self, x, training=False):
        x = self.s1_0(x)
        x = self.s1_1(x)
        x = self.s1_2(x)
        x = self.s1_3(x)

        #Stage 2

        x = self.s2_0(x)
        x = self.s2_1(x)
        x = self.s2_2(x)

        # Stage 3
        x = self.s3_0(x)
        x = self.s3_1(x)
        x = self.s3_2(x)
        x = self.s3_3(x)

        # Stage 4
        x = self.s4_0(x)
        x = self.s4_1(x)
        x = self.s4_2(x)
        x = self.s4_3(x)
        x = self.s4_4(x)
        x = self.s4_5(x)

        
        # Stage 5
        x = self.s5_0(x)
        x = self.s5_1(x)
        x = self.s5_2(x)
        x = self.s5_3(x)

        #Final Stage
        x = self.sf_0(x)
        x = self.sf_1(x)
        x = self.sf_2(x)

        return x


class EncoderBase(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_0 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='conv_0')
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
        self.dense_0 = layers.Dense(units=2048, activation='relu')
        # self.do_0 = layers.Dropout(rate=dropout_rate)

        self.dense_1 =layers.Dense(units=512, activation='relu')
        # self.do_1 =layers.Dropout(rate=dropout_rate)


        self.dense_2 =layers.Dense(units=2048, activation='relu')

    def call(self, x, training=False):
        x = self.dense_0(x)
        # x = self.do_0(x, training=training)
        #
        x = self.dense_1(x)
        # x = self.do_1(x,training=training)

        x = self.dense_2(x)

        return x


class ClassifierHead(layers.Layer):
    def __init__(self, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dense_0 = layers.Dense(units=2048, activation='relu')
        self.do_0 = layers.Dropout(rate=dropout_rate)

        self.dense_1 =layers.Dense(units=128, activation='relu')

        self.do_1 =layers.Dropout(rate=dropout_rate)

        self.dense_2 = layers.Dense(units=128, activation='relu')
        self.dense_3 = layers.Dense(units=128, activation='relu')
        self.dense_4 = layers.Dense(units=128, activation='relu')
        self.dense_5 = layers.Dense(units=128, activation='relu')
        self.dense_6 = layers.Dense(units=128, activation='relu')
        self.dense_7 =layers.Dense(units=1)

    def call(self, x, training=False):
        x = self.dense_0(x)
        # x = self.do_0(x, training=training)

        # x = self.dense_1(x)
        # # x = self.do_1(x,training=training)
        # x = self.dense_2(x)
        # x = self.dense_3(x)
        # x = self.dense_4(x)
        # x = self.dense_5(x)
        # x = self.dense_6(x)
        x = self.dense_7(x)

        return x

class SimSiameseBasedModel(Model):
    def __init__(self):
        super(SimSiameseBasedModel, self).__init__()

        self.train_contrastive = True

        # self.encoder_base = EncoderBase()
        # self.encoder_base = ResNet50()
        # self.encoder_base =  tf.keras.Sequential([hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/035-128-feature-vector/versions/2",
        #            trainable=True)])
        self.classifier_head = ClassifierHead(dropout_rate=0.05)
        self.probe = ProbeHead()

        self.normalizer_audio_0 = tf.keras.layers.Normalization(axis=[1,2])
        self.normalizer_audio_1 = tf.keras.layers.Normalization(axis=[1,2])
        self.normalizer_mos = tf.keras.layers.Normalization(axis=1)

        self.classifier_loss = keras.losses.MeanSquaredError()

        self.lambda_ = .9

        # Contrastive Parameters
        self.temperature = 0.1

    def compile(self, opt_ctr=keras.optimizers.Adam(learning_rate=1e-3), opt_cls=keras.optimizers.Adam(learning_rate=1e-3), **kwargs):
        super().compile(**kwargs)

        self.opt_contrastive = opt_ctr
        self.opt_classifier = opt_cls

        self.loss_classifier_metric = keras.metrics.MeanSquaredError(name="loss")
        self.loss_contrastive = keras.metrics.Mean(name="loss_ctr")

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        mos = tf.reshape(data['mos'], [-1])
        audio_0 = data['ref_f_abs']
        audio_1 = data['deg_f_abs']
        audios = tf.stack([audio_0, audio_1], axis=3)

        # audio_0 = self.normalizer_audio_0(audio_0)
        # audio_1 = self.normalizer_audio_0(audio_1)
        # mos = self.normalizer_mos(mos)
        # if self.train_contrastive:
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            # Compute the loss value
            # (the loss function is configured in `compile()`
            noise_std_audio = 0.01
            # audio_0_n = audio_0 + tf.random.normal(tf.shape(audio_0), mean=0, stddev=noise_std_audio)
            audio_0_n = audio_0 + tf.random.normal(tf.shape(audio_0), mean=0, stddev=noise_std_audio)
            audio_1_n = audio_1 + tf.random.normal(tf.shape(audio_1), mean=0, stddev=noise_std_audio)

            # audio_0_n = tf.expand_dims(audio_0_n, axis=3)
            # audio_1_n = tf.expand_dims(audio_1_n, axis=3)

            p_0, z_0 = self([audio_0_n,audio_1_n], contrastive=True, training=True)
            p_1, z_1 = self([audio_0_n,audio_1_n], contrastive=True, training=True)

            # noise_std_mos = 0.05
            # mos_n = mos + tf.random.normal(tf.shape(mos), mean=0, stddev=noise_std_mos)
            mos_n = mos

            loss_0 = self.SimSiameseLoss(p_0, z_1, mos_n, mos_n, 0.01, 1)
            loss_1 = self.SimSiameseLoss(p_1, z_0, mos_n, mos_n, 0.01, 1)
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

            pred = self([audio_0, audio_1], contrastive = False, training=True)

            loss = self.classifier_loss(mos, pred)

        # Compute gradients
        gradients = tape.gradient(
            loss,
            self.classifier_head.trainable_weights+self.encoder_base.trainable_weights
        )

        self.opt_classifier.apply_gradients(
            zip(
                gradients,
                self.classifier_head.trainable_weights+self.encoder_base.trainable_weights
            )
        )
        # Update weights
        # Update metrics (includes the metric that tracks the loss)
        # Return a dict mapping metric names to current value
        self.loss_classifier_metric.update_state(mos, pred)
        # self.loss_contrastive.update_state(loss_cont)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        mos = tf.reshape(data['mos'], [-1])
        audio_0 = data['ref_f_abs']
        audio_1 = data['deg_f_abs']
        audios = tf.stack([audio_0, audio_1], axis=3)

        # audio_0 = self.normalizer_audio_0(audio_0)
        # audio_1 = self.normalizer_audio_0(audio_1)
        # mos = self.normalizer_mos(mos)

        # audio_0 = tf.expand_dims(audio_0, axis=3)
        # audio_1 = tf.expand_dims(audio_1, axis=3)

        pred = self([audio_0, audio_1], contrastive=False, training=False)
        loss = self.classifier_loss(mos, pred)

        self.loss_classifier_metric.update_state(mos, pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, data_inp, training=False, contrastive=False):
        if contrastive:
            audio_0, audio_1 = data_inp
            audios = tf.stack([audio_0, audio_1], axis=3)
            p = self.encoder_base(audios)
            z = self.probe(p, training=training)
            return p, z
        else:
            audio_0, audio_1 = data_inp
            audios = tf.stack([audio_0, audio_1], axis=3)

            # x_0 = self.encoder_base(audio_0)
            x_1 = self.encoder_base(audios)
            # x = tf.concat([x_1, x_1], axis=1)

            x = self.classifier_head(x_1, training=training)
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

        # loss_p = cond_p*tf.math.abs(m_diff)*projection
        # loss_n = cond_n*m_diff*projection
        # loss_p = tf.math.exp(-loss_p/tf.reduce_sum(projection, keepdims=True))
        # loss_n = tf.math.exp(-loss_n / tf.reduce_sum(projection, keepdims=True))
        # loss_p = tf.math.log(tf.reduce_sum(loss_p))
        # loss_n = tf.math.log(tf.reduce_sum(loss_n))
        # loss = loss_p - lambda_*loss_n
        # # loss = tf.reduce_sum(loss,axis=1)
        # loss = -tf.reduce_mean(loss)

        B = BATCH_SIZE
        y = tf.range(B)
        cce = tf.keras.losses.SparseCategoricalCrossentropy ()

        loss = cce(y, projection)

        return loss
