import torch

pretrained_pth = './pretrained/audiotransformer_mini_mAP_4896.pt'
   
dump = torch.load(pretrained_pth, map_location='cpu')
if 'model' in dump:
    dump = dump['model']
    
for k, v in dump.items():
    if 'init_bn' in k or 'freq_pos_embed' in k:
        print(k)
        print(v.shape)
    if k == 'init_bn.1.num_batches_tracked':
        print(v)