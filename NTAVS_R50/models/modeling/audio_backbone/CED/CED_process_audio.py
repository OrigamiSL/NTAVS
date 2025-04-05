from functools import partial
import math
from typing import Any, Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torchaudio.transforms as audio_transforms
import torchaudio
import resampy
import numpy as np
class FrontEnd(nn.Sequential):

    def __init__(self,
                 f_min: int = 0,
                 sample_rate: int = 16000,
                 win_size: int = 512,
                 center: bool = True,
                 n_fft: int = 512,
                 f_max: Optional[int] = None,
                 hop_size: int = 160,
                 n_mels: int = 64):
        self.f_min = f_min
        self.sample_rate = sample_rate
        self.win_size = win_size
        self.center = center
        self.n_fft = n_fft
        self.f_max = f_max
        self.hop_size = hop_size
        self.n_mels = n_mels

        super().__init__(
            audio_transforms.MelSpectrogram(f_min=self.f_min,
                                            sample_rate=self.sample_rate,
                                            win_length=self.win_size,
                                            center=self.center,
                                            n_fft=self.n_fft,
                                            f_max=self.f_max,
                                            hop_length=self.hop_size,
                                            n_mels=self.n_mels),
            audio_transforms.AmplitudeToDB(top_db=120))

    # Disable Autocast for FP16 training!
    @autocast(enabled=False)
    def forward(self, x):
        return super().forward(x)

def get_CED_process(wavepath, mel_bins): 
    get_mel = FrontEnd(f_min=0,
                        f_max=8000,
                        center= True,
                        win_size= 512,
                        hop_size= 160,
                        sample_rate=16000,
                        n_fft=512,
                        n_mels= mel_bins)
    data, sr = torchaudio.load(wavepath)
    wav_data, sr = sf.read(wav_file, dtype="int16")
    # print('wav_data, sr', wav_data.shape, sr) (80320, 2) 16000
    assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
    samples = wav_data / 32768.0
    print(type(data))
    if len(data.shape) > 1:
        data = torch.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sr != 16000:
        data = resampy.resample(data, sr, 16000)
    log_mel_examples = get_mel(data)

    print(log_mel_examples.shape)
    log_mel_examples = log_mel_examples[:, None, :, :].float()

    return log_mel_examples