import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from utils import *
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam,calc_ssim
from data_loader import build_datasets
from models.WMIRNet import WMIRNet
from validate import validate
from train import train
import pdb
import args_parser
from torch.nn import functional as F
import cv2
from time import *
import os
import scipy.io as io
from thop import profile
torch.cuda.is_available()
import matplotlib.pyplot  as plt 

def visualize_bands(img_hsi, band_indices, max_val=1.0):
    
    img_hsi = np.clip(img_hsi, 0, max_val)
    
    
    img_rgb = img_hsi[:, :, band_indices]
    
    
    img_rgb_8bit = (img_rgb / max_val * 255.0).astype(np.uint8)
    
    
    img_bgr_8bit = cv2.cvtColor(img_rgb_8bit, cv2.COLOR_RGB2BGR)
    
    return img_bgr_8bit

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)


def main():
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    elif args.dataset == 'Pavia':
      args.n_bands = 102
    elif args.dataset == 'IEEE2018':
      args.n_bands = 48
   
      

    train_list, test_list = build_datasets(args.root, 
                                           args.dataset, 
                                           args.image_size, 
                                           args.n_select_bands, 
                                           args.scale_ratio)
    

    if args.arch == 'WMIRNet':
      model = WMIRNet(
        img_size=args.image_size,       
        patch_size=1,
        in_chans_MSI=args.n_select_bands,
        in_chans_HSI=args.n_bands,
        embed_dim=128,                
        depths=(1,),                     
        d_state=32,                     
        drop_rate=0.,
        mlp_ratio=2.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,            
        upscale=args.scale_ratio,       
        img_range=1.,
        upsampler='',                    
        resi_connection='1conv'
    ).cuda()
    
        

    model_path = f"./checkpoints/{args.dataset}_{args.arch}_{int(args.scale_ratio)}.pkl"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))

    test_ref, test_lr, test_hr = test_list
    model.eval()


    ref = test_ref.float().detach()
    lr = test_lr.float().detach()
    hr = test_hr.float().detach()
    
    begin_time = time()
    if args.arch == 'SSRNET':
        out, _, _, _, _, _ = model(lr.cuda(), hr.cuda())
    elif args.arch == 'SpatRNET':
        _, out, _, _, _, _ = model(lr.cuda(), hr.cuda())
    elif args.arch == 'SpecRNET':
        _, _, out, _, _, _ = model(lr.cuda(), hr.cuda())
    elif args.arch == 'SwinCGAN':
        out, _, _, _, _, _ = model(lr.cuda(), hr.cuda(), args.scale_ratio)
    else:
        out, _, _, _, _, _ = model(lr.cuda(), hr.cuda())
    end_time = time()
    run_time = (end_time-begin_time)*1000

    
    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    
    slr  =  F.interpolate(lr, scale_factor=args.scale_ratio, mode='bilinear')
    slr = slr.detach().cpu().numpy()
    slr  =  np.squeeze(slr).transpose(1,2,0).astype(np.float64)
    
    sref = np.squeeze(ref).transpose(1,2,0).astype(np.float64)
    sout = np.squeeze(out).transpose(1,2,0).astype(np.float64)

    


   
    

    H, W, C = sref.shape
 
    VIS_BANDS = [C - 3, C - 2, C - 1]
     


   
    MAX_PIXEL_VALUE = np.max(sref) 
    

    vis_ref = visualize_bands(sref, VIS_BANDS, MAX_PIXEL_VALUE)
    vis_out = visualize_bands(sout, VIS_BANDS, MAX_PIXEL_VALUE)
    vis_slr = visualize_bands(slr, VIS_BANDS, MAX_PIXEL_VALUE)

 
    vis_dir = os.path.join('./实验结果的mat格式', args.dataset, str(args.scale_ratio)+'倍', 'VISUAL_OUTPUT')
    os.makedirs(vis_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(vis_dir, 'GT_Reference.png'), vis_ref)
    cv2.imwrite(os.path.join(vis_dir, f'{args.arch}_Output.png'), vis_out)
    cv2.imwrite(os.path.join(vis_dir, 'Baseline_Bilinear.png'), vis_slr)

    
    results_dir = './实验结果的mat格式/' + args.dataset + '/' + str(args.scale_ratio) + '倍/'
    os.makedirs(results_dir, exist_ok=True)


    traditional_dir = './为传统方法准备数据/' + args.dataset + '/' + str(args.scale_ratio) + '倍/'
    os.makedirs(traditional_dir, exist_ok=True)
    

    io.savemat('./实验结果的mat格式/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+args.arch+'.mat',{'Out':sout})

    io.savemat('./实验结果的mat格式/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'REF.mat',{'REF':sref})

    io.savemat('./实验结果的mat格式/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'Upsample.mat',{'Out':slr})
    
    t_lr = np.squeeze(lr).detach().cpu().numpy().transpose(1,2,0).astype(np.float64)
    t_hr = np.squeeze(hr).detach().cpu().numpy().transpose(1,2,0).astype(np.float64)
    
    io.savemat('./为传统方法准备数据/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'lr'+'.mat',{'HSI':t_lr})
    io.savemat('./为传统方法准备数据/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'hr'+'.mat',{'MSI':t_hr})
    
    
    psnr = calc_psnr(ref, out)
    rmse = calc_rmse(ref, out)
    ergas = calc_ergas(ref, out)
    sam = calc_sam(ref, out)
    ssim = calc_ssim(ref, out)


   
if __name__ == '__main__':
    main()
