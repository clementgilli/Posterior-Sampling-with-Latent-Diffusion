import torch
from torch.fft import fft2, ifft2
import numpy as np

def box_inpainting(x, box_size=128):
    x = x.clone()
    _, _, h, w = x.shape
    x[:, :, h//2-box_size//2:h//2+box_size//2, w//2-box_size//2:w//2+box_size//2] = 0
    return x

def random_inpainting(x, prob=(0.2, 0.8)):
    x = x.clone()
    drop_prob = (torch.rand(1, device=x.device) * (prob[1] - prob[0]) + prob[0]).item()
    print(drop_prob)
    mask = torch.bernoulli(torch.full_like(x, 1 - drop_prob))
    x = x * mask
    return x

def super_resolution(x, scale_factor=4):
    x = x.clone()
    x = torch.nn.functional.interpolate(x, scale_factor=1/scale_factor, mode='nearest', antialias=False)
    return x

def gaussian_blur(x, sigma=2.0):
    x = x.clone()
    size = 25
    z = torch.arange(size) - size//2
    kt = torch.exp(-0.5 * (z**2 + z.view(-1, 1)**2) / sigma**2)
    
    (m,n) = kt.shape
    M,N = 256,256

    k = torch.zeros((M,N))
    k[0:m,0:n] = kt/torch.sum(kt)
    k = torch.roll(k,(-int(m/2),-int(n/2)),(0,1))
    fk = fft2(k).to(x.device)
    return ifft2(fk*fft2(x)).real

def blur_from_file(x, filename):
    x = x.clone()
    kt = torch.tensor(np.loadtxt('kernels/'+filename))
    
    (m,n) = kt.shape
    M,N = 256,256

    k = torch.zeros((M,N))
    k[0:m,0:n] = kt/torch.sum(kt)
    k = torch.roll(k,(-int(m/2),-int(n/2)),(0,1))
    fk = fft2(k).to(x.device)
    
    (m,n) = kt.shape
    M,N = 256,256

    k = torch.zeros((M,N))
    k[0:m,0:n] = kt/torch.sum(kt)
    k = torch.roll(k,(-int(m/2),-int(n/2)),(0,1))
    fk = fft2(k).to(x.device)
    return ifft2(fk*fft2(x)).real

def linear_operator(x, mode, nu):
    if "box_inpainting" in mode:
        y = box_inpainting(x)
        return y + nu * torch.randn_like(y)
    elif "random_inpainting" in mode:
        y = random_inpainting(x)
        return y + nu * torch.randn_like(y)
    elif "super_resolution" in mode:
        scale_factor = int(mode.split(":")[1]) if ":" in mode else 4
        y = super_resolution(x, scale_factor)
        return y + nu * torch.randn_like(y)
    elif "gaussian_blur" in mode:
        sigma = int(mode.split(":")[1]) if ":" in mode else 2
        y = gaussian_blur(x, sigma)
        return y + nu * torch.randn_like(y)
    elif "blur_from_file" in mode:
        filename = mode.split(":")[1] if ":" in mode else "kernel1.txt"
        y = blur_from_file(x, filename)
        return y + nu * torch.randn_like(y)
    else:
        raise ValueError("Unknown mode")