# link: https://github.com/facebookresearch/detectron2/blob/80307d2d5e06f06a8a677cc2653f23a4c56402ac/detectron2/data/datasets/cityscapes.py
import csv
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


logger = logging.getLogger(__name__)


def _get_avsms3_files(image_dir, gt_dir, audio_dir, pre_mask_dir=None,
                      audio1_dir= None, audio2_dir= None, audio3_dir=None):
    files = []
    # scan through the directory
    videos = PathManager.ls(gt_dir)
    logger.info(f"{len(videos)} videos found in '{gt_dir}'.")
    videos = sorted(videos)
    for video in videos:
        if video == '.DS_Store' or video == '._.DS_Store':
            continue
        video_img_dir = os.path.join(image_dir, video)
        video_gt_dir = os.path.join(gt_dir, video)
        video_pre_mask_dir = os.path.join(pre_mask_dir, video) if pre_mask_dir else None
        basenames = PathManager.ls(video_img_dir)

        image_files = []
        label_files = []
        pre_mask_files = []
        basenames = sorted(basenames)

        audio_file = os.path.join(audio_dir, basenames[0][:-10] + ".pkl")
       
        audio2_file = os.path.join(audio2_dir, basenames[0][:-10] + ".pkl")
        audio3_file = os.path.join(audio3_dir, basenames[0][:-10] + ".pkl")
        for num_img, basename in enumerate(basenames):
            image_file = os.path.join(video_img_dir, basename)
            pre_mask_file = os.path.join(video_pre_mask_dir, basename.replace(".png", "_mask_color.png")) if video_pre_mask_dir else None
            suffix = ".png"
            assert basename.endswith(suffix), basename  # * assert that the file is a png file
            image_files.append(image_file)
            pre_mask_files.append(pre_mask_file)
        
        gt_basenames = PathManager.ls(video_gt_dir)
        gt_basenames = sorted(gt_basenames)
        for num_img, basename in enumerate(gt_basenames):
            label_file = os.path.join(video_gt_dir, basename)
            label_files.append(label_file)

        files.append((image_files, label_files, audio_file, pre_mask_files,
                      audio2_file, audio3_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f[0] if isinstance(f, list) else f), f
    return files


def load_avsms3_semantic(img_dir, mask_dir, audio_dir, pre_mask_dir=None, split="train",
                         audio1_dir= None, audio2_dir= None, audio3_dir= None):
    """
    Args:
        img_dir (str): path to the image directory.
        mask_dir (str): path to the mask directory.
        audio_dir (str): path to the audio directory.
        pre_mask_dir (str): path to the pre_mask directory.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    
    # get the (img_dir, mask_dir, audio_dir, pre_mask_dir) files name
    files = _get_avsms3_files(img_dir, mask_dir, audio_dir, pre_mask_dir
                              ,audio1_dir, audio2_dir, audio3_dir)

    for image_file, label_file, audio_file, pre_mask_file, \
        audio2_file, audio3_file in files:
        if pre_mask_file[0] is None:
            ret.append(
                {
                    "file_names": image_file,
                    "sem_seg_file_names": label_file,
                    "audio_file_name": audio_file,
                    "audio2_file": audio2_file, 
                    "audio3_file": audio3_file
                }
            )
        else:
            ret.append(
                {
                    "file_names": image_file,
                    "sem_seg_file_names": label_file,
                    "audio_file_name": audio_file,
                    "pre_mask_file_names": pre_mask_file,
                    "audio2_file": audio2_file, 
                    "audio3_file": audio3_file
                }
            )

    assert PathManager.isfile(ret[0]["sem_seg_file_names"][0]), ret[0]["sem_seg_file_names"][0]
    return ret

def register_avsms3_semantic(root):
    meta_data = os.path.join(root, "ms3_meta_data.csv")
    root = os.path.join(root, "ms3_data")

    for name, dirname in [("train", "train"), ("val", "val"), ("test", "test")]:
        image_dir = os.path.join(root, "visual_frames")
        gt_dir = os.path.join(root, "gt_masks", dirname)

        audio_dir = os.path.join(root, "audio_wav_512_96_new_scale", dirname)

        pre_mask_dir = os.path.join(root, "pre_SAM_mask")
        name = f"avsms3_sem_seg_{name}"

       
        audio_scale2_dir = os.path.join(root, "audio_wav_256_96_new_scale", dirname)
        audio_scale3_dir = os.path.join(root, "audio_wav_1024_96_new_scale", dirname)

        logger.info(f"register DatasetCatalog of dataset: '{name}'.")
        # register the dataset
        DatasetCatalog.register(
            name,
            lambda x=image_dir, y=gt_dir, z=audio_dir, pre_mask_dir=pre_mask_dir, 
            s1 = None, s2 = audio_scale2_dir, s3 = audio_scale3_dir,
            split=dirname,: load_avsms3_semantic(
                img_dir=x, mask_dir=y, audio_dir=z, pre_mask_dir=pre_mask_dir, split=split,
                audio1_dir= s1, audio2_dir= s2, audio3_dir= s3,
            ),
        )
        logger.info(f"register MetadataCatalog of dataset: '{name}'.")
        MetadataCatalog.get(name).set(
            stuff_classes=["background", "object"],
            # stuff_colors=[(0, 0, 0), (255, 255, 255)],
            evaluator_type="sem_seg",
            ignore_label=255,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
print('dataset_root', _root)
if _root.endswith("/Multi-sources/"):
    logger.info(f"dataset root is '{_root}'.")
    register_avsms3_semantic(_root)
