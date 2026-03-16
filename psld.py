import os
import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, VQModel
from tqdm import tqdm

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

    loss = loss_per_image.sum()
    grad = torch.autograd.grad(loss, z)[0]

    zeta = args.zeta_scale / torch.sqrt(loss_per_image + 1e-8).view(B, 1, 1, 1)

    if args.sampler == "ddpm":
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
        
        z_next = z_prev - zeta * grad
        
    else:
        raise ValueError("Sampler inconnu")

    return z_next


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="box_inpainting", 
                        choices=["identity", "box_inpainting", "random_inpainting", "super_resolution", "gaussian_blur", "blur_from_file"])
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"], help="Sampler choice")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--zeta_scale", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--gluing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process at once")
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    
    args = parser.parse_args()

    torch.manual_seed(6)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using : {device}")

    os.makedirs("results", exist_ok=True)

    vqvae = VQModel.from_pretrained("./models/vqvae", torch_dtype=torch.float32).to(device)
    unet = UNet2DModel.from_pretrained("./models/unet", torch_dtype=torch.float32).to(device)
    
    vqvae.eval()
    unet.eval()

    if args.sampler == "ddpm":
        scheduler = DDPMScheduler.from_pretrained("./models/scheduler")
    else:
        scheduler = DDIMScheduler.from_pretrained("./models/scheduler")
    scheduler.set_timesteps(args.steps)

    x0_list = []
    for idx in [1, 59, 462, 478]:##range(args.batch_size):
        img_path = f'ffhq256-1k-validation/{str(idx).zfill(5)}.png' 
        x0_list.append(im2tensor(plt.imread(img_path), device=device))
        
    x_true = torch.cat(x0_list, dim=0) # (B, 3, 256, 256)
    B = x_true.shape[0]
    imgshape = x_true.shape
    imgshape_latent = (B, unet.config.in_channels, unet.sample_size, unet.sample_size)

    operator = LinearOperator(args.mode, imgshape, device)
    evaluator = ImageMetrics(device=device)

    y = operator.measure(x_true, nu=args.nu)
    
    for i in range(B):
        torchvision.utils.save_image(x_true[i] * 0.5 + 0.5, f"results/orig_{i}.png")
        torchvision.utils.save_image(y[i] * 0.5 + 0.5, f"results/degraded_{i}.png")

    alphas = scheduler.alphas.to(device) if args.sampler == "ddpm" else None
    betas = scheduler.betas.to(device) if args.sampler == "ddpm" else None
    alphas_bar = scheduler.alphas_cumprod.to(device)

    z = torch.randn(imgshape_latent, device=device)

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        
        # for DDIM
        prev_t = scheduler.timesteps[i + 1] if i < len(scheduler.timesteps) - 1 else torch.tensor(-1, device=device)
        t_tensor = torch.full((B,), t.item(), device=device, dtype=torch.long)
        
        z = z.detach().requires_grad_(True)

        with torch.amp.autocast("cuda"):
        
            s_residus = unet(z, t_tensor)["sample"]
            z0_hat =  (z - torch.sqrt(1.0 - alphas_bar[t]) * s_residus) / torch.sqrt(alphas_bar[t])
            x0_hat = vqvae.decode(z0_hat)[0]

        z = perform_one_step(z, t, prev_t, s_residus, z0_hat, x0_hat, y, operator, vqvae, args, alphas, betas, alphas_bar)


    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            final_img = vqvae.decode(z.detach())[0]

    for i in range(B):
        torchvision.utils.save_image(final_img[i] * 0.5 + 0.5, f"results/recon_{i}.png")

    results = evaluator.evaluate_all(x_true, final_img, data_range=2.0)
    
    metrics_path = f"results/metrics_{args.mode}_{args.sampler}.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Configuration : Mode={args.mode}, Sampler={args.sampler}, Steps={args.steps}, Batch={B}\n")
        f.write("-" * 50 + "\n")
        f.write(f"PSNR  : {results['PSNR']:.4f} dB\n")
        f.write(f"SSIM  : {results['SSIM']:.4f}\n")
        f.write(f"LPIPS : {results['LPIPS']:.4f}\n")
        
    print(f"Done.")

if __name__ == "__main__":
    main()