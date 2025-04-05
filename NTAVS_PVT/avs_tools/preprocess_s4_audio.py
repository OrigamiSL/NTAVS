# Copyright (c) Facebook, Inc. and its affiliates.
# link: https://github.com/facebookresearch/detectron2/blob/80307d2d5e06f06a8a677cc2653f23a4c56402ac/detectron2/data/datasets/cityscapes.py
import functools
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import torch
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image
import pandas as pd
import pickle
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from tqdm import tqdm
import sys; sys.path.append(os.getcwd())
from models.modeling.audio_backbone.torchvggish import vggish_input

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

logger = logging.getLogger(__name__)
def _get_avss_files(df_split, root, split, f_size, t_size):
    files = []

    basic_root = os.path.join(root, 's4_data/audio_wav'+'_'+str(f_size)+'_'+str(t_size)+'_'+'new_scale')
    os.makedirs(basic_root, exist_ok= True)
    for index in tqdm(np.arange(0,len(df_split))):
        df_one_video = df_split.iloc[index]
        video_name, set, category = df_one_video['name'], df_one_video['split'], df_one_video['category']
        os.makedirs(os.path.join(basic_root, set, category), exist_ok= True)
       
        audio_path = os.path.join(root, 's4_data/audio_wav', set, category, video_name+'.wav')
        audio_sacle_path = os.path.join(basic_root, set, category, video_name+'.pkl')

        x = vggish_input.wavfile_to_examples(audio_path, f_size = f_size, t_size = t_size)

        if x.shape[0] != 5:
            N_SECONDS, CHANNEL, N_BINS, N_BANDS = x.shape
            new_lm_tensor = torch.zeros(5, CHANNEL, N_BINS, N_BANDS)
            new_lm_tensor[:N_SECONDS] = x
            new_lm_tensor[N_SECONDS:] = x[-1].repeat(5-N_SECONDS, 1, 1, 1)
            x = new_lm_tensor

        with open(audio_sacle_path, "wb") as fw:
            pickle.dump(x, fw)
        # files.append((audio_path, vid_temporal_mask_flag, gt_temporal_mask_flag))
    
    return files

def load_avss_semantic(df_split, root, split, f_size, t_size):
    """
        Args:
            img_dir (str): path to the image directory.
            mask_dir (str): path to the mask directory.
        Returns:
            list[dict]: a list of dicts in Detectron2 standard format. each has "file_name" and
                "sem_seg_file_name".
    """
    ret = []

    files = _get_avss_files(df_split, root, split, f_size = f_size, t_size = t_size)
      
    # print('len(files): ', len(files))


if __name__=="__main__":

    img_dir = './AVS_dataset/AVSBench_object/Single-source'
    splits = ["train","val","test"]
    df_all = pd.read_csv(os.path.join(img_dir,'s4_meta_data.csv'), sep=',')
    ft = [(1024, 96), (512, 96), (256, 96)]
    
    for f, t in ft:
        for split in splits:
            df_split = df_all[df_all['split'] == split]
            ret = load_avss_semantic(df_split=df_split, root=img_dir, split=split, f_size = f, t_size = t)
    