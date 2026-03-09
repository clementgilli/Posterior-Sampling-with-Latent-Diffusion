import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
import warnings

class ImageMetrics:
    def __init__(self, device=None, net='alex'):
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lpips_model = lpips.LPIPS(net=net).to(self.device)
            self.lpips_model.eval()

    def _format_tensor(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
            
        if img.dim() == 3:
            img = img.unsqueeze(0)
            
        return img

    def compute_psnr(self, img_true, img_gen, data_range=2.0):
        img_true = self._format_tensor(img_true)
        img_gen = self._format_tensor(img_gen)
        
        mse = torch.mean((img_true - img_gen) ** 2).item()
        if mse == 0:
            return float('inf')
            
        rmse = np.sqrt(mse)
        return 20 * np.log10(data_range / rmse)

    def compute_ssim(self, img_true, img_gen, data_range=2.0):
        img_true = self._format_tensor(img_true).detach().cpu().numpy()
        img_gen = self._format_tensor(img_gen).detach().cpu().numpy()
        
        batch_size = img_true.shape[0]
        ssim_total = 0.0
        
        for i in range(batch_size):
            ssim_total += ssim(
                img_true[i], 
                img_gen[i], 
                channel_axis=0, 
                data_range=data_range
            )
            
        return ssim_total / batch_size

    def compute_lpips(self, img_true, img_gen):
        img_true = self._format_tensor(img_true).to(self.device)
        img_gen = self._format_tensor(img_gen).to(self.device)
        
        with torch.no_grad():
            score = self.lpips_model(img_true, img_gen).mean().item()
            
        return score

    def evaluate_all(self, img_true, img_gen, data_range=2.0):
        
        return {
            "PSNR": self.compute_psnr(img_true, img_gen, data_range),
            "SSIM": self.compute_ssim(img_true, img_gen, data_range),
            "LPIPS": self.compute_lpips(img_true, img_gen)
        }