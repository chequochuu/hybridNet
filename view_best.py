import os
import torch
from torchsummary import summary
bestdir ='Data/best/'
dirs = os.listdir(bestdir)
for i in dirs:
    t = torch.load(bestdir+i)
    para = i.split('--')
    for (j) in (para):
        if (j.find('batch_size')!=-1):
            batch_size = j.split('_')[-2] 
    print(t['score'])
    N = t['Net']
    summary(N, (1,1024), int(batch_size))
    
    
