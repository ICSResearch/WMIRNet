import torch
import numpy as np
import math
import torch.nn.functional as F

def calc_ergas(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/4*ergas**0.5

    return ergas

def calc_psnr(img_tgt, img_fus):
    mse = np.mean((img_tgt-img_fus)**2)
    img_max = np.max(img_tgt)
    psnr = 10*np.log10(img_max**2/mse)

    return psnr

def calc_rmse(img_tgt, img_fus):
    rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))

    return rmse


def calc_sam(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt / np.max(img_tgt)
    img_fus = img_fus / np.max(img_fus)

    A = np.sqrt(np.sum(img_tgt**2, axis=0))
    B = np.sqrt(np.sum(img_fus**2, axis=0))
    AB = np.sum(img_tgt*img_fus, axis=0)

    sam = AB/(A*B)
    sam = np.arccos(sam)
    sam = np.mean(sam)*180/3.1415926535

    return sam




def gaussian(window_size, sigma, device=None, dtype=torch.float32):
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
                         dtype=dtype, device=device)
    return (gauss / gauss.sum()).unsqueeze(1)  

def create_window(window_size, sigma, channel, device=None, dtype=torch.float32):
    _1D = gaussian(window_size, sigma, device=device, dtype=dtype)
    _2D = (_1D @ _1D.t()).unsqueeze(0).unsqueeze(0)  
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _to_torch_nchw(x):
    """Convert numpy or torch array to torch tensor in NCHW float format (values not rescaled)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not torch.is_tensor(x):
        raise TypeError("Input must be numpy array or torch tensor")
    
    orig_dtype = x.dtype
   
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
 
    if x.dim() == 2: 
        x = x.unsqueeze(0).unsqueeze(0)  
    elif x.dim() == 3:  
        a, b, c = x.shape
        
        if c <= min(a, b):
            
            x = x.permute(2, 0, 1).unsqueeze(0)
        else:
            
            x = x.unsqueeze(0)
    elif x.dim() == 4:
        pass  
    else:
        raise ValueError("Unsupported tensor shape")
    return x

def calc_ssim(img_tgt, img_fus, data_range=None, window_size=11,
              sigma=None, return_map=False, return_channelwise=False, eps=1e-12):
    xt = _to_torch_nchw(img_tgt)
    xf = _to_torch_nchw(img_fus)

    if xt.shape != xf.shape:
        raise ValueError(f"Input shapes must match after conversion to NCHW, got {xt.shape} vs {xf.shape}")

    device = xt.device
    dtype = xt.dtype

    N, C, H, W = xt.shape
    ws = min(window_size, H, W)
    if ws % 2 == 0:
        ws = ws - 1 if ws > 1 else 1
    if sigma is None:
        sigma = 1.5 * ws / 11

    
    if data_range is None:
        max_val = float(max(xt.max().item(), xf.max().item()))
        data_range = 1.0 if max_val <= 1.01 else 255.0

   
    window = create_window(ws, sigma, C, device=device, dtype=dtype)

    
    mu1 = F.conv2d(xt, window, padding=ws // 2, groups=C)
    mu2 = F.conv2d(xf, window, padding=ws // 2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(xt * xt, window, padding=ws // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(xf * xf, window, padding=ws // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(xt * xf, window, padding=ws // 2, groups=C) - mu1_mu2

    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    
    numerator = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + eps)

    
    if return_map:
        return ssim_map.detach().cpu().numpy()  

 
    ssim_per_channel = ssim_map.mean(dim=(2, 3)).mean(dim=0)  
    if return_channelwise:
        return ssim_per_channel.detach().cpu().numpy()

   
    return float(ssim_per_channel.mean().item())

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6  
