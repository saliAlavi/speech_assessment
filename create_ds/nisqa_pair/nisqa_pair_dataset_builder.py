"""nisqa dataset."""

import tensorflow_datasets as tfds
import numpy as np
import scipy.io
import pandas as pd
import librosa
import dataclasses
import os
from typing import Tuple


PATH_DS_BASE = '/fs/scratch/PAS2622/Project_AI/Datasets/NISQA_Corpus'

@dataclasses.dataclass
class BuilderConfigEEG(tfds.core.BuilderConfig):
    sampling_rate: int = 16000
    hop_len: int = 160
    win_len: int = 512
    n_mels: int = 128
    fixed_len: int = 12
    normalize: bool = False

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for nisqa dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
      BuilderConfigEEG(name='mos_pairs', description='signal and the stft', sampling_rate = 16000,),

  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(nisqa): Specifies the tfds.core.DatasetInfo object

    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # 'ref_wav': tfds.features.Tensor(dtype=np.float32, shape=(self._builder_config.sampling_rate*self._builder_config.fixed_len,)),
            # 'deg_wav': tfds.features.Tensor(dtype=np.float32, shape=(self._builder_config.sampling_rate*self._builder_config.fixed_len,)),
            'ref_f_abs_0': tfds.features.Tensor(dtype=np.float32, shape=(1+int(self._builder_config.win_len//2),1+int(self._builder_config.sampling_rate*self._builder_config.fixed_len//self._builder_config.hop_len))),
            'ref_f_abs_1': tfds.features.Tensor(dtype=np.float32, shape=(1+int(self._builder_config.win_len//2),1+int(self._builder_config.sampling_rate*self._builder_config.fixed_len//self._builder_config.hop_len))),
            # 'ref_f_phase': tfds.features.Tensor(dtype=np.float32, shape=(1+int(self._builder_config.win_len//2),1+int(self._builder_config.sampling_rate*self._builder_config.fixed_len//self._builder_config.hop_len))),
            # 'ref_len': tfds.features.Tensor(dtype=np.float32, shape=(1,)), #shape=(1,)),
            'deg_f_abs_0': tfds.features.Tensor(dtype=np.float32, shape=(1+int(self._builder_config.win_len//2),1+int(self._builder_config.sampling_rate*self._builder_config.fixed_len//self._builder_config.hop_len))),
            'deg_f_abs_1': tfds.features.Tensor(dtype=np.float32, shape=(1+int(self._builder_config.win_len//2),1+int(self._builder_config.sampling_rate*self._builder_config.fixed_len//self._builder_config.hop_len))),
            # 'deg_f_phase': tfds.features.Tensor(dtype=np.float32, shape=(1+int(self._builder_config.win_len//2),1+int(self._builder_config.sampling_rate*self._builder_config.fixed_len//self._builder_config.hop_len))),
            # 'deg_len':  tfds.features.Tensor(dtype=np.float32,shape=(1,)), #shape=(1,)),
            'log_mel_spectogram_0': tfds.features.Tensor(dtype=np.float32, shape=(self._builder_config.n_mels ,1 + int(self._builder_config.sampling_rate * self._builder_config.fixed_len // self._builder_config.hop_len))),
            'log_mel_spectogram_1': tfds.features.Tensor(dtype=np.float32, shape=(self._builder_config.n_mels ,1 + int(self._builder_config.sampling_rate * self._builder_config.fixed_len // self._builder_config.hop_len))),
            'mos':      tfds.features.Tensor(dtype=np.float32,shape=(1,)) #shape=(1,)),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  @staticmethod
  def get_stft(wav, sr, fixed_len, hop_len, win_len, normalize=False):
    wav_length = wav.shape[0]
    n_fft = win_len
    wav_pad = librosa.util.fix_length(data=wav, size=sr * fixed_len)  # signal_length + n_fft // 2)

    f_wav = librosa.stft(wav_pad, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window=scipy.signal.hamming)

    f_abs = np.abs(f_wav)
    f_phase = np.angle(f_wav)

    if normalize == True:
      meanR = np.mean(f_abs, axis=1).reshape((257, 1))
      stdR = np.std(f_abs, axis=1).reshape((257, 1)) + 1e-12
      f_abs = (f_abs - meanR) / stdR

    return wav_pad, f_abs, f_phase, wav_length

  @staticmethod
  def get_melspectrum(wav, sr, fixed_len, hop_len, win_len, n_mels  ):
      wav_length = wav.shape[0]
      n_fft = win_len
      wav_pad = librosa.util.fix_length(data=wav, size=sr * fixed_len)
      mel_spectrogram = librosa.feature.melspectrogram(y=wav_pad, sr=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
      log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

      return wav_pad, log_mel_spectrogram, wav_length

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(nisqa): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(nisqa): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train_sim': self._generate_examples('train_sim'),
        'train_live': self._generate_examples('train_live'),
        'test_sim': self._generate_examples('test_sim'),
        'test_live': self._generate_examples('test_live')
    }

  def _generate_examples(self, split):
    """Yields examples."""
    sampling_rate = self._builder_config.sampling_rate
    hop_len = self._builder_config.hop_len
    win_len = self._builder_config.win_len
    fixed_len = self._builder_config.fixed_len
    normalize = self._builder_config.normalize
    n_mels = self._builder_config.n_mels

    corpus_path = PATH_DS_BASE + '/'
    df_train_sim = pd.read_csv(corpus_path + 'NISQA_TRAIN_SIM/NISQA_TRAIN_SIM_file.csv')
    df_train_live = pd.read_csv(corpus_path + 'NISQA_TRAIN_LIVE/NISQA_TRAIN_LIVE_file.csv')

    df_train = pd.concat([df_train_sim, df_train_live], join='inner')

    df_valid_sim = pd.read_csv(corpus_path + 'NISQA_VAL_SIM/NISQA_VAL_SIM_file.csv')
    df_valid_live = pd.read_csv(corpus_path + 'NISQA_VAL_LIVE/NISQA_VAL_LIVE_file.csv')

    df_valid = pd.concat([df_valid_sim, df_valid_live], join='inner')

    if split=='train_sim':
        df = df_train_sim
    elif split=='train_live':
        df = df_train_live
    elif split=='test_sim':
        df = df_valid_sim
    elif split=='test_live':
        df=df_valid_live

    df.sort_values(['mos'], inplace=True)
    df.round({'mos': 1})
    
    for i in range(len(df)-1):
        if (df.iloc[i]['mos'] == df.iloc[i+1]['mos']):
            line_0, line_1 = df.iloc[i], df.iloc[i+1]
            
            path_ref_0 = os.path.join(PATH_DS_BASE,line_0.filepath_ref)
            path_deg_0 = os.path.join(PATH_DS_BASE,line_0.filepath_deg)

            path_ref_1 = os.path.join(PATH_DS_BASE,line_1.filepath_ref)
            path_deg_1 = os.path.join(PATH_DS_BASE,line_1.filepath_deg)

            mos     = np.asarray(line_0.mos)

            ref_wav_0, _ = librosa.load(path_ref_0, sr=sampling_rate)
            deg_wav_0, _ = librosa.load(path_deg_0, sr=sampling_rate)

            ref_wav_1, _ = librosa.load(path_ref_1, sr=sampling_rate)
            deg_wav_1, _ = librosa.load(path_deg_1, sr=sampling_rate)

            ref_wav_padded_0, ref_f_abs_0, ref_f_phase_0, ref_len_0 = self.get_stft(ref_wav_0, sampling_rate, fixed_len, hop_len, win_len, normalize)
            deg_deg_padded_0, deg_f_abs_0, deg_f_phase_0, deg_len_0 = self.get_stft(deg_wav_0, sampling_rate, fixed_len, hop_len, win_len, normalize)

            ref_wav_padded_1, ref_f_abs_1, ref_f_phase_1, ref_len_1 = self.get_stft(ref_wav_1, sampling_rate, fixed_len, hop_len, win_len, normalize)
            deg_deg_padded_1, deg_f_abs_1, deg_f_phase_1, deg_len_1 = self.get_stft(deg_wav_1, sampling_rate, fixed_len, hop_len, win_len, normalize)

            _, log_mel_spectrogram_0, _ = self.get_melspectrum(deg_wav_0, sampling_rate, fixed_len, hop_len, win_len, n_mels)
            _, log_mel_spectrogram_1, _ = self.get_melspectrum(deg_wav_1, sampling_rate, fixed_len, hop_len, win_len, n_mels)

            # ref_len= np.asarray(ref_len).reshape((1,)).astype(np.float32)
            # deg_len= np.asarray(deg_len).reshape((1,)).astype(np.float32)
            
            yield i, {
                # 'ref_wav': ref_wav_padded,
                # 'deg_wav': deg_deg_padded,
                'ref_f_abs_0':ref_f_abs_0,
                'ref_f_abs_1':ref_f_abs_1,
                # 'ref_f_phase':ref_f_phase,
                # 'ref_len':  ref_len.reshape((1,)),
                'deg_f_abs_0':deg_f_abs_0,
                'deg_f_abs_1':deg_f_abs_1,
                # 'deg_f_phase':deg_f_phase,
                # 'deg_len':deg_len,
                'log_mel_spectogram_0': log_mel_spectrogram_0,
                'log_mel_spectogram_1': log_mel_spectrogram_1,

                'mos': mos.reshape((1,)).astype(np.float32)
            }
