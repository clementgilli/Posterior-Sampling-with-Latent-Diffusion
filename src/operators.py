import torch
from torch.fft import fft2, ifft2
import numpy as np

class LinearOperator:
    def __init__(self, mode, imgshape, device, box_size=128, prob=(0.2, 0.8)):
        
        self.mode = mode
        self.device = device
        self.imgshape = imgshape
        _, _, self.h, self.w = imgshape
        
        if "identity" in mode:
            pass
        elif "box_inpainting" in mode:
            if ":" in mode:
                box_size = int(mode.split(":")[1])
            self.mask = torch.ones(imgshape, device=device)
            self.mask[:, :, self.h//2-box_size//2:self.h//2+box_size//2, self.w//2-box_size//2:self.w//2+box_size//2] = 0
            
        elif "random_inpainting" in mode:
            drop_prob = (torch.rand(1, device=device) * (prob[1] - prob[0]) + prob[0]).item()
            print(f"Random inpainting drop probability: {drop_prob:.3f}")
            self.mask = torch.bernoulli(torch.full(imgshape, 1 - drop_prob, device=device))
            
        elif "super_resolution" in mode:
            self.scale_factor = int(mode.split(":")[1]) if ":" in mode else 4
            
        elif "gaussian_blur" in mode:
            sigma = float(mode.split(":")[1]) if ":" in mode else 2.0
            size = 25
            z = torch.arange(size, device=device) - size//2
            kt = torch.exp(-0.5 * (z**2 + z.view(-1, 1)**2) / sigma**2)
            
            m, n = kt.shape
            k = torch.zeros((self.h, self.w), device=device)
            k[0:m, 0:n] = kt / torch.sum(kt)
            k = torch.roll(k, (-int(m/2), -int(n/2)), (0, 1))
            self.fk = fft2(k)
            
        elif "blur_from_file" in mode:
            filename = mode.split(":")[1] if ":" in mode else "kernel1.txt"
            kt = torch.tensor(np.loadtxt('kernels/'+filename), dtype=torch.float32, device=device)
            
            m, n = kt.shape
            k = torch.zeros((self.h, self.w), device=device)
            k[0:m, 0:n] = kt / torch.sum(kt)
            k = torch.roll(k, (-int(m/2), -int(n/2)), (0, 1))
            self.fk = fft2(k)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x):
        if "indentity" in self.mode:
          return x
        if "inpainting" in self.mode:
            return x * self.mask
        elif "super_resolution" in self.mode:
            return torch.nn.functional.interpolate(x, scale_factor=1/self.scale_factor, mode='nearest', antialias=False)
        elif "blur" in self.mode:
            return ifft2(self.fk * fft2(x)).real
        return x

    def measure(self, x, nu=0.0):
        
        y = self.forward(x)
        if nu > 0:
            y = y + nu * torch.randn_like(y)
        return y
    
    def visualize_y(self, y):
        if "super_resolution" in self.mode:
            return torch.nn.functional.interpolate(y, scale_factor=self.scale_factor, mode='nearest', antialias=False)
        return y
    
    def transpose(self, y):
        if "identity" in self.mode:
            return y

        if "inpainting" in self.mode:
            # L'opérateur est diagonal (matrice identité masquée). A^T = A.
            return y * self.mask
            
        elif "blur" in self.mode:
            # L'adjoint d'une convolution spatiale est la convolution avec le noyau symétrique.
            # Dans le domaine fréquentiel, cela revient à multiplier par le conjugué complexe.
            return ifft2(torch.conj(self.fk) * fft2(y)).real
            
        elif "super_resolution" in self.mode:
            # Solution mathématiquement exacte (VJP) pour la transposée de n'importe quel 
            # opérateur linéaire PyTorch (y compris l'interpolation bilinéaire).
            x_dummy = torch.zeros(self.imgshape, device=self.device, requires_grad=True)
            Ax = self.forward(x_dummy)
            return torch.autograd.grad(Ax, x_dummy, grad_outputs=y)[0]
            
        return y