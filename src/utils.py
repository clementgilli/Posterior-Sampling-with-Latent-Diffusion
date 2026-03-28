import torch
import numpy as np
import matplotlib.pyplot as plt
import IPython.display
import tempfile

def im2tensor(x, device):
    x = torch.tensor(x,device=device)
    x = 2*x-1 # [0,1]->[-1,1]   
    return x.permute(2,0,1).unsqueeze(0)

def tensor2im(x):
    x = 0.5+0.5*x # [-1,1]->[0,1]
    return x.detach().cpu().permute(2,3,1,0).squeeze()
    
def viewimage(im, normalize=True,vmin=0,vmax=1,titre='',displayfilename=False):
    im = tensor2im(im)
    imin= im.numpy().astype(np.float32)
    
    if normalize:
        if vmin is None:
            vmin = imin.min()
        if vmax is None:
            vmax = imin.max()
        if np.abs(vmax-vmin)>1e-10:
            imin = (imin.clip(vmin,vmax)-vmin)/(vmax-vmin)
        else:
            imin = vmin
    else:
        imin=imin.clip(0,255)/255
    imin=(imin*255).astype(np.uint8)
    filename=tempfile.mktemp(titre+'.png')
    if displayfilename:
        print (filename)
    plt.imsave(filename, imin, cmap='gray')
    IPython.display.display(IPython.display.Image(filename))