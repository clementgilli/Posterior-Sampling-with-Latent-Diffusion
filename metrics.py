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
        
        mse = torch.mean((img_true - img_gen) ** 2, axis=(1, 2, 3))
        rmse = torch.sqrt(mse)
        psnr = 20 * torch.log10(data_range / rmse)
        return psnr.mean().item(), psnr.std().item()

    def compute_ssim(self, img_true, img_gen, data_range=2.0):
        img_true = self._format_tensor(img_true).detach().cpu().numpy()
        img_gen = self._format_tensor(img_gen).detach().cpu().numpy()
        
        batch_size = img_true.shape[0]
        ssim_total = []
        
        for i in range(batch_size):
            ssim_total += [ssim(
                img_true[i], 
                img_gen[i], 
                channel_axis=0, 
                data_range=data_range
            )]
        ssim_total = torch.tensor(ssim_total)
            
        return ssim_total.mean().item(), ssim_total.std().item()

    def compute_lpips(self, img_true, img_gen):
        img_true = self._format_tensor(img_true).to(self.device)
        img_gen = self._format_tensor(img_gen).to(self.device)
        
        with torch.no_grad():
            score = self.lpips_model(img_true, img_gen)
            
        return score.mean().item(), score.std().item()

    def evaluate_all(self, img_true, img_gen, data_range=2.0):
        mean_psnr, std_psnr = self.compute_psnr(img_true, img_gen, data_range)
        mean_ssim, std_ssim = self.compute_ssim(img_true, img_gen, data_range)
        mean_lpips, std_lpips = self.compute_lpips(img_true, img_gen)
        return {
            "PSNR": mean_psnr,
            "SSIM": mean_ssim,
            "LPIPS": mean_lpips,
            "PSNR_STD": std_psnr,
            "SSIM_STD": std_ssim,
            "LPIPS_STD": std_lpips
        }