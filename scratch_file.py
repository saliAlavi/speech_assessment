import pprint
import tempfile

from IPython import display
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_models as tfm

# These are not in the tfm public API for v2.9. They will be available in v2.10
from official.vision.serving import export_saved_model_lib
import official.core.train_lib


exp_config = tfm.core.exp_factory.get_exp_config('resnet_imagenet')
tfds_name = 'cifar10'
ds,ds_info = tfds.load(
tfds_name,
with_info=True)
ds_info

