## NTAVS
This repository provides the PyTorch implementation for the paper "Nontrivial Audio-Visual Segmentation", which is submitted to Knowledge-Based Systems (KBS).

## Abstract
In mathematics, the adjective `trivial' is frequently used to refer to an object with a simple structure. Although it is counterintuitive to imagine that a multimodal task such as audio-visual segmentation (AVS) can be trivial, the trivial acoustic features possessed by the existing AVS models tremendously limit their performance. Indeed, acoustic features are normally averaged and repeated in the time-frequency domain to fit the sizes of visual features used in previous approaches. Thus, the triviality of these acoustic features degrades the effectiveness of audio-visual fusion and segmentation. In contrast, a novel global temporal audio-visual mixer (GTAVM) is proposed in this work; this approach fuses globally the temporal dependencies of audio and visual features along the input video sequence, transforming the average of the acoustic features into the average of the global temporal relations. Therefore, the acoustic features are able to maintain their sizes and generate nontrivial audio queries for segmentation purposes. To further enrich the acoustic features, a hierarchical audio backbone, in which multiple time-frequency resolutions are utilized to generate hierarchical Mel-spectrograms as audio inputs, is proposed in this work. Extensive experiments conducted on three AVS benchmarks demonstrate the state-of-the-art performance of the proposed methods.

## Method
<p align="center">
  <img  src="image/framework.jpg">

<h6 align="center">Overview of the proposed NTAVS.</h6>
</p>

## Preparation
This work contains two versions, i.e., NTAVS-PVT and NTAVS-R50, which employ two different visual backbones. We implement the two verisions in the `./NTAVS_PVT` and `./NTAVS_R50`, respectively. For loading the datasets and the pretrained weights conviently, you can run the script below to creat symbolic links for the `./NTAVS_PVT` and `./NTAVS_R50` folders. 
```
sh ./creat_symbolic_link.sh
```
You can choose both or one of them to train and test. Take the NTAVS-PVT as an example, you should firstly run the command:
```
cd ./NTAVS_PVT
```
We follow [COMBO](https://yannqi.github.io/AVS-COMBO) to finish the preparation before training and testing the models. The details are as follows:

### 1. Environments

- Linux or macOS with Python ≥ 3.6

```shell
# creat the conda environment
conda env create -f NTAVS.yaml
# activate
conda activate NTAVS
# build MSDeformAttention
cd models/modeling/pixel_decoder/ops
sh make.sh
```
- Preprocessing for detectron2

  For using Siam-Encoder Module (SEM), we refine 1-line code of the detectron2.

  The refined file that requires attention is located at:

  `conda_envs/xxx/lib/python3.xx/site-packages/detectron2/checkpoint/c2_model_loading.py`
  (refine the `xxx`  to your own environment)

  Commenting out the following code in [L287](https://github.com/facebookresearch/detectron2/blob/cc9266c2396d5545315e3601027ba4bc28e8c95b/detectron2/checkpoint/c2_model_loading.py#L287) will allow the code to run without errors:

```python
# raise ValueError("Cannot match one checkpoint key to multiple keys in the model.")  
```

- Install Semantic-SAM (Optional)

```shell
# Semantic-SAM
pip install git+https://github.com/cocodataset/panopticapi.git
git clone https://github.com/UX-Decoder/Semantic-SAM
cd Semantic-SAM
python -m pip install -r requirements.txt
```
Find out more at [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM)

### 2. Datasets

Please refer to the link [AVSBenchmark](https://github.com/OpenNLPLab/AVSBench) to download the datasets. You need to put the data under the `../AVS_dataset`. The folder tree shall look like:

```
|--AVS_dataset
   |--AVSBench_semantic/
   |--AVSBench_object/Multi-sources/
   |--AVSBench_object/Single-source/
```

Then run the scripts below to preprocess the AVSS dataset for efficient training.

```shell
python3 avs_tools/preprocess_avss_audio.py
python3 avs_tools/process_avssimg2fixsize.py
```
To generate the proposed MSAI in this work, you should run the scripts below:
```
python ./avs_tools/preprocess_avss_audio.py  # for AVSS dataset
python ./avs_tools/preprocess_s3_audio.py # for MS3 dataset
python ./avs_tools/preprocess_s4_audio.py # for S4 dataset
```
Then you can obtain the file tree:
```
|--AVS_dataset
     |--AVSBench_object/Multi-sources/ms3_data
       |--audio_wav_256_96_new_scale
       |--audio_wav_512_96_new_scale
       |--audio_wav_1024_96_new_scale
    |--AVSBench_object/Single-source/s4_data
       |--audio_wav_256_96_new_scale
       |--audio_wav_512_96_new_scale
       |--audio_wav_1024_96_new_scale
    |--AVSBench_semantic
       |--v1m
          |--_19NVGk6Zt8_0
             |--audio_wav_256_96_new_scale.pkl
             |--audio_wav_512_96_new_scale.pkl
             |--audio_wav_1024_96_new_scale.pkl
    ...
```

### 3. Download Pre-Trained Models

- The pretrained visual backbone (ResNet-50 and PVT-v2) is available from benchmark AVSBench pretrained backbones [YannQi/COMBO-AVS-checkpoints · Hugging Face](https://huggingface.co/YannQi/COMBO-AVS-checkpoints).
- The pretrained acoustic backbone (CED-Mini) is available from [CED-Mini](https://huggingface.co/mispeech/ced-mini).

After you finish downloading, put the weights under the `../pretrained`. 

```
|--pretrained
   |--detectron2/R-50.pkl
   |--detectron2/d2_pvt_v2_b5.pkl
   |--audiotransformer_mini_mAP_4896.pt
```

### 4. Maskiges pregeneration

- Generate class-agnostic masks (Optional)

```shell
sh avs_tools/pre_mask/pre_mask_semantic_sam_s4.sh train # or ms3, avss
sh avs_tools/pre_mask/pre_mask_semantic_sam_s4.sh val 
sh avs_tools/pre_mask/pre_mask_semantic_sam_s4.sh test
```

- Generate Maskiges (Optional)

```shell
python3 avs_tools/pre_mask2rgb/mask_precess_s4.py --split train # or ms3, avss
python3 avs_tools/pre_mask2rgb/mask_precess_s4.py --split val
python3 avs_tools/pre_mask2rgb/mask_precess_s4.py --split test
```

- Move Maskiges to the following folder
  Note: For convenience, the pre-generated Maskiges for S4\MS3\AVSS subset can be obtained at [YannQi/COMBO-AVS-checkpoints · Hugging Face](https://huggingface.co/YannQi/COMBO-AVS-checkpoints).

The file tree shall look like:
```
|--AVS_dataset
    |--AVSBench_semantic/pre_SAM_mask/
    |--AVSBench_object/Multi-sources/ms3_data/pre_SAM_mask/
    |--AVSBench_object/Single-source/s4_data/pre_SAM_mask/
```

## Train and Test
To record the training epoch wherein the model achieves the best performance, we add several lines of codes in the `conda_envs/xxx/lib/python3.xx/site-packages/detectron2/engine/hooks.py`.
The codes are located in the function named `after_step` (lines 547–560), as shown below:
```
def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0 :
            with open ('iter.txt', 'w') as f:
                f.write(str(self.trainer.iter))
                f.close()
            if next_iter != self.trainer.max_iter:
                self._do_eval()
```
The you can start to train and test. The scripts for training and testing are put under `./scripts`.
### 1. Train

```shell
# ResNet-50 (Attention! The scripts below are under `./NTAVS_R50` folder.)
sh scripts/res_train_avs4.sh
sh scripts/res_train_avms3.sh
sh scripts/res_train_avss.sh
```

```shell
# PVTv2 (Attention! The scripts below are under `./NTAVS_PVT` folder.)
sh scripts/pvt_train_avs4.sh
sh scripts/pvt_train_avms3.sh
sh scripts/pvt_train_avss.sh
```

### 2. Test
After you finish the training process, you can evaluate the best checkpoints by the commands below:
```shell
# ResNet-50 (Attention! The scripts below are under `./NTAVS_R50` folder.)
sh scripts/res_test_avs4.sh
sh scripts/res_test_avms3.sh
sh scripts/res_test_avss.sh
```

```shell
# PVTv2 (Attention! The scripts below are under `./NTAVS_PVT` folder.)
sh scripts/pvt_test_avs4.sh
sh scripts/pvt_test_avms3.sh
sh scripts/pvt_test_avss.sh
```

## Acknowledgement

This codebase is implemented on the following project. We really appreciate its authors for the open-source works!
- [COMBO](https://github.com/yannqi/COMBO-AVS) [[related paper](https://arxiv.org/pdf/2312.06462)]


**This project is not for commercial use. For commercial use, please contact the author.**

## Citation

If any part of our work helps your research, please consider citing us and giving a star to our repository.

