your_dataset_root='/home/lhg/work/ssd_new/AVSBench_all/COMBO-AVS-main/AVS_dataset'
your_pretrained_path='/home/lhg/work/ssd_new/AVSBench_all/COMBO-AVS-main/pretrained'

ln -s "$your_dataset_root" ./NTAVS_R50

ln -s "$your_dataset_root" ./NTAVS_PVT

ln -s "$your_pretrained_path" ./NTAVS_R50

ln -s "$your_pretrained_path" ./NTAVS_PVT
