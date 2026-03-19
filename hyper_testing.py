import os
import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, VQModel
from tqdm import tqdm
import datetime
import numpy as np
import gc

from utils import im2tensor
from operators import LinearOperator
from metrics import ImageMetrics

def perform_one_step(z, t, prev_t, s_residus, z0_hat, x0_hat, y, operator, vqvae, args, alphas, betas, alphas_bar):
    B = z.shape[0]
    
    resid = operator.forward(x0_hat) - y
    loss_likelihood_per_image = torch.linalg.norm(resid.reshape(B, -1), dim=1)

    if args.gluing:
        ortho_project = x0_hat - operator.transpose(operator.forward(x0_hat))
        parallel_project = operator.transpose(y)
        gluing_image = (parallel_project + ortho_project).clamp(-1.0, 1.0)

        encoded_z_0 = vqvae.encode(gluing_image).latents
        loss_glue_per_image = torch.linalg.norm((encoded_z_0 - z0_hat).reshape(B, -1), dim=1)

        loss_per_image = args.eta * loss_likelihood_per_image + args.gamma * loss_glue_per_image
    else:
        loss_per_image = args.eta * loss_likelihood_per_image

    # Summing allows independent gradients per image in the batch
    loss = loss_per_image.sum()
    grad = torch.autograd.grad(loss, z)[0]

    if args.sampler == "ddpm":
        zeta = args.zeta_scale / torch.sqrt(loss_per_image + 1e-8).view(B, 1, 1, 1)
        z_prim = (z - (betas[t] / torch.sqrt(1.0 - alphas_bar[t])) * s_residus) / torch.sqrt(alphas[t])
        eps = torch.sqrt(betas[t]) * torch.randn_like(z) if t > 0 else 0
        z_next = z_prim + eps - zeta * grad

    elif args.sampler == "ddim":
        alpha_bar_t = alphas_bar[t]
        alpha_bar_prev = alphas_bar[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=z.device)
        
        variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
        sigma_t = args.ddim_eta * torch.sqrt(variance)
        
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)) * s_residus
        
        noise = torch.randn_like(z) if prev_t >= 0 else 0.0
        z_prev = torch.sqrt(alpha_bar_prev) * z0_hat + dir_xt + sigma_t * noise
        
        scaling = 1000.0 / args.steps
        zeta = scaling * torch.sqrt(alpha_bar_t)
        
        z_next = z_prev - zeta * grad
        
    else:
        raise ValueError("Unknown sampler")

    return z_next


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="box_inpainting", 
                        choices=["identity", "box_inpainting", "random_inpainting", "super_resolution", "gaussian_blur", "blur_from_file"])
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=1.0) # Likelihood weight
    parser.add_argument("--gamma", type=float, default=0.1) # Gluing weight
    parser.add_argument("--zeta_scale", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--gluing", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--precise_mode", type=str, default=None)
    parser.add_argument("--num_batchs", type=int, default=1)
    
    # NEW ARGUMENTS FOR HYPERPARAMETER SWEEPING
    parser.add_argument("--sweep_param", type=str, default="gamma", help="Which parameter to sweep (e.g., gamma, eta, nu)")
    parser.add_argument("--range_min", type=float, default=0.01)
    parser.add_argument("--range_max", type=float, default=1.0)
    parser.add_argument("--num_points", type=int, default=10, help="Number of steps between min and max")
    parser.add_argument("--log_scale", action="store_true", help="Use log scale for the sweep array")

    args = parser.parse_args()

    torch.manual_seed(6)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using : {device}")

    vqvae = VQModel.from_pretrained("./models/vqvae", torch_dtype=torch.float32).to(device)
    unet = UNet2DModel.from_pretrained("./models/unet", torch_dtype=torch.float32).to(device)
    vqvae.eval()
    unet.eval()

    if args.sampler == "ddpm":
        scheduler = DDPMScheduler.from_pretrained("./models/scheduler")
    else:
        scheduler = DDIMScheduler.from_pretrained("./models/scheduler")
    scheduler.set_timesteps(args.steps)

    evaluator = ImageMetrics(device=device)

    if args.precise_mode is not None:
        args.mode = args.mode + ":" + args.precise_mode

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/{args.mode}_{args.sweep_param}_sweep_{datetime_str}"
    os.makedirs(save_path, exist_ok=True)

    # Generate Sweep Array
    if args.log_scale:
        param_values = np.logspace(np.log10(args.range_min), np.log10(args.range_max), args.num_points)
    else:
        param_values = np.linspace(args.range_min, args.range_max, args.num_points)

    # Initialize CSV
    csv_file = f"{save_path}/metrics_sweep.csv"
    with open(csv_file, "w") as f:
        f.write(f"{args.sweep_param},PSNR_mean,PSNR_std,SSIM_mean,SSIM_std,LPIPS_mean,LPIPS_std\n")

    print(f"Sweeping {args.sweep_param} from {args.range_min} to {args.range_max} ({args.num_points} steps)")

    # OUTER LOOP: Hyperparameter Sweep
    for p_val in param_values:
        setattr(args, args.sweep_param, p_val)
        print(f"\n=====================================")
        print(f"Testing {args.sweep_param} = {p_val:.4f}")
        print(f"=====================================")

        all_psnr, all_ssim, all_lpips = [], [], []

        # INNER LOOP: Batch Processing
        for batch_idx in range(args.num_batchs):
            
            x0_list = []
            for idx in range(batch_idx * args.batch_size, (batch_idx + 1) * args.batch_size): 
                img_path = f'ffhq256-1k-validation/{str(idx).zfill(5)}.png' 
                x0_list.append(im2tensor(plt.imread(img_path), device=device))
                
            x_true = torch.cat(x0_list, dim=0)
            B = x_true.shape[0]
            imgshape = x_true.shape
            imgshape_latent = (B, unet.config.in_channels, unet.sample_size, unet.sample_size)

            operator = LinearOperator(args.mode, imgshape, device)
            y = operator.measure(x_true, nu=args.nu)
            
            # Save Orig/Degraded only on the first sweep to save disk space
            if p_val == param_values[0]:
                for i in range(B):
                    global_i = i + batch_idx * args.batch_size
                    torchvision.utils.save_image(x_true[i] * 0.5 + 0.5, f"{save_path}/orig_{global_i}.png")
                    torchvision.utils.save_image(y[i] * 0.5 + 0.5, f"{save_path}/degraded_{global_i}.png")

            alphas = scheduler.alphas.to(device) if args.sampler == "ddpm" else None
            betas = scheduler.betas.to(device) if args.sampler == "ddpm" else None
            alphas_bar = scheduler.alphas_cumprod.to(device)

            z = torch.randn(imgshape_latent, device=device)

            for i, t in enumerate(tqdm(scheduler.timesteps, desc=f"Batch {batch_idx+1}/{args.num_batchs}")):
                prev_t = scheduler.timesteps[i + 1] if i < len(scheduler.timesteps) - 1 else torch.tensor(-1, device=device)
                t_tensor = torch.full((B,), t.item(), device=device, dtype=torch.long)
                
                # STRICT STOP-GRADIENT ON U-NET
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        s_residus = unet(z.detach(), t_tensor)["sample"].detach()

                # ENABLE GRADIENTS FOR ANALYTICAL MATH
                z = z.detach().requires_grad_(True)

                with torch.amp.autocast("cuda"):
                    z0_hat = (z - torch.sqrt(1.0 - alphas_bar[t]) * s_residus) / torch.sqrt(alphas_bar[t])
                    x0_hat = vqvae.decode(z0_hat)[0]

                z = perform_one_step(z, t, prev_t, s_residus, z0_hat, x0_hat, y, operator, vqvae, args, alphas, betas, alphas_bar)

            # EVALUATION
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    final_img = vqvae.decode(z.detach())[0]

            for i in range(B):
                global_i = i + batch_idx * args.batch_size
                torchvision.utils.save_image(final_img[i] * 0.5 + 0.5, f"{save_path}/recon_param_{p_val:.4f}_{global_i}.png")
                
                res = evaluator.evaluate_all(x_true[i:i+1], final_img[i:i+1], data_range=2.0)
                # Ensure we pull floats so we don't leak VGG memory
                all_psnr.append(res['PSNR'] if isinstance(res['PSNR'], float) else res['PSNR'].item())
                all_ssim.append(res['SSIM'] if isinstance(res['SSIM'], float) else res['SSIM'].item())
                all_lpips.append(res['LPIPS'] if isinstance(res['LPIPS'], float) else res['LPIPS'].item())

            # Clear memory per batch
            del z, s_residus, x0_hat, z0_hat, final_img, x_true, y
            torch.cuda.empty_cache()
            gc.collect()

        # Aggregation for this parameter
        results = {
            'PSNR': np.mean(all_psnr), 'PSNR_STD': np.std(all_psnr),
            'SSIM': np.mean(all_ssim), 'SSIM_STD': np.std(all_ssim),
            'LPIPS': np.mean(all_lpips), 'LPIPS_STD': np.std(all_lpips)
        }
        
        # Write to CSV incrementally
        with open(csv_file, "a") as f:
            f.write(f"{p_val:.6f},{results['PSNR']:.4f},{results['PSNR_STD']:.4f},{results['SSIM']:.4f},{results['SSIM_STD']:.4f},{results['LPIPS']:.4f},{results['LPIPS_STD']:.4f}\n")
        
        print(f"Results for {args.sweep_param} = {p_val:.4f}:")
        print(f"PSNR: {results['PSNR']:.4f}  |  SSIM: {results['SSIM']:.4f}  |  LPIPS: {results['LPIPS']:.4f}")

    print(f"\nGrid search complete. Results saved in {csv_file}")

if __name__ == "__main__":
    main()