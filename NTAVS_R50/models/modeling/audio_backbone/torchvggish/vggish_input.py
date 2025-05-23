# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

# Modification: Return torch tensors rather than numpy arrays
import torch
import os
import numpy as np
import resampy

from . import mel_features
from . import vggish_params

import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def waveform_to_examples(data, sample_rate, return_tensor=True, if_return_length = False,
                         f_size = None, t_size = None, if_return_stft = False):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
      data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
      sample_rate: Sample rate of data.
      return_tensor: Return data as a Pytorch tensor ready for VGGish

    Returns:
      3-D np.array of shape [num_examples, num_frames, num_bands] which represents
      a sequence of examples, each of which contains a patch of log mel
      spectrogram, covering num_frames frames of audio and num_bands mel frequency
      bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.

    """
    # Convert to mono.
    # print(f_size, t_size)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE: # 16000
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    # print('data', data.shape) (80320,), (160000,)
    # Compute log mel spectrogram features.
    if f_size == None:
       log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE, # 16000
        log_offset=vggish_params.LOG_OFFSET, # 0.01
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS, #0.025
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS, # 0.01
        num_mel_bins=vggish_params.NUM_MEL_BINS, # 64
        lower_edge_hertz=vggish_params.MEL_MIN_HZ, #125
        upper_edge_hertz=vggish_params.MEL_MAX_HZ, #7500
    )
    else:
      if if_return_stft:
         stft_draw = mel_features.stft_spectrogram(data,
            audio_sample_rate=vggish_params.SAMPLE_RATE, # 16000
            log_offset=vggish_params.LOG_OFFSET, # 0.01
            window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS, #0.025
            hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS, # 0.01
            num_mel_bins=vggish_params.NUM_MEL_BINS, # 64
            window_length_new = f_size,
            lower_edge_hertz=0, #125
            upper_edge_hertz=8000)
         return stft_draw
      else:
        log_mel = mel_features.log_mel_spectrogram(
            data,
            audio_sample_rate=vggish_params.SAMPLE_RATE, # 16000
            log_offset=vggish_params.LOG_OFFSET, # 0.01
            window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS, #0.025
            hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS, # 0.01
            num_mel_bins=vggish_params.NUM_MEL_BINS, # 64
            window_length_new = f_size,
            # num_mel_bins= f_size,
            lower_edge_hertz=0, #125
            upper_edge_hertz=8000, #7500
      )

    # Frame features into examples.
    # print('log_mel', log_mel.shape) (500, 64)
    if f_size == None:
      features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS # 1 /0.01 = 100
      example_window_length = int(round(vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate)) # 0.96 * 100 = 96
      example_hop_length = int(round(vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate)) # 96
    else:
      example_window_length = t_size
      example_hop_length = t_size
      
    log_mel_examples = mel_features.frame(log_mel, window_length=example_window_length, hop_length=example_hop_length)
    print('log_mel_examples', log_mel_examples.shape)

    if return_tensor:
        log_mel_examples = torch.tensor(log_mel_examples, requires_grad=True)[:, None, :, :].float()

    if if_return_length:
      return log_mel_examples, example_hop_length
    else:
      return log_mel_examples


def wavfile_to_examples(wav_file, return_tensor=True, f_size = None, t_size = None):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.

    Args:
      wav_file: String path to a file, or a file-like object. The file
      is assumed to contain WAV audio data with signed 16-bit PCM samples.
      torch: Return data as a Pytorch tensor ready for VGGish

    Returns:
      See waveform_to_examples.
    """
    wav_data, sr = sf.read(wav_file, dtype="int16")
    # print('wav_data, sr', wav_data.shape, sr) (80320, 2) 16000
    assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples(samples, sr, return_tensor, f_size = f_size, t_size = t_size)

def draw_wavfile(wav_file, return_tensor=False):

    # save_path = './audio.png'
    wav_data, sr = sf.read(wav_file, dtype="int16")
    assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    mel, hop_length = waveform_to_examples(samples, sr, return_tensor = False, if_return_length= True)
    # mel = mel[0]
    save_path_dir = './fig/audio/'+wav_file.split('/')[-1]
    os.makedirs(save_path_dir, exist_ok= True)
    for i in range(len(mel)):
      librosa.display.specshow(mel[i], x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length)
      # plt.colorbar(format="%+2.0f dB")
      plt.axis('off')
      plt.savefig(os.path.join(save_path_dir, str(i)+'.jpg'), bbox_inches = 'tight', pad_inches = 0.1, transparent = True)
      plt.close()
    return 

def draw_stft(wav_file, f_size):
  wav_data, sr = sf.read(wav_file, dtype="int16")
  assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
  samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
  stft_spec = waveform_to_examples(samples, sr, return_tensor = False, 
                                    if_return_length= True, f_size = f_size, if_return_stft= True)
  t, f = stft_spec.shape
  x_d = np.array([i for i in range(t)])
  y_d = np.array([i for i in range(f)])
  X, Y = np.meshgrid(x_d, y_d)
  # stft_spec = stft_spec.transpose(1,0)
  X = X.transpose(1, 0)
  Y = Y.transpose(1, 0)
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  # print(X.shape, Y.shape, stft_spec.shape)

  # ax.view_init(elev=0,
  #              azim=0)
  
  ax.plot_surface(X, Y, stft_spec, rstride=1, cstride=1, cmap='rainbow')
  # ax = Axes3D(fig)
  # ax.scatter(X, Y, stft_spec, c='k', s=1)
  ax.grid(True)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.set_zticklabels([])
  # ax.set_xticks([])
  # ax.set_yticks([])
  # ax.set_zticks([])
  # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  # ax.w_zaxis.line.set_visible(False)
  # plt.grid()
  # print(stft_spec.shape)
  save_path_dir = './fig/stft/'+wav_file.split('/')[-1]
  os.makedirs(save_path_dir, exist_ok= True)

  save_plt_path = os.path.join(save_path_dir, str(f)+'.jpg')
  # plt.title('Level 1')
  plt.savefig(save_plt_path, bbox_inches = 'tight', transparent = True, dpi = 1000)
  plt.close()

