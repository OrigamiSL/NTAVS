your_dataset_root='./AVS_dataset'
your_pretrained_path='./pretrained'

ln -s "$your_dataset_root" ./NTAVS_R50

ln -s "$your_dataset_root" ./NTAVS_PVT

ln -s "$your_pretrained_path" ./NTAVS_R50

ln -s "$your_pretrained_path" ./NTAVS_PVT
