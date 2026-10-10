import time
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np


from models.WMIRNet import WMIRNet 
from data_loader import build_datasets
from utils import *
from validate import validate
from train import train
import args_parser




def main():
    args = args_parser.args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if args.dataset == 'PaviaU':
        args.n_bands = 103
    elif args.dataset == 'Pavia':
        args.n_bands = 102
    elif args.dataset == 'IEEE2018':
        args.n_bands = 48
    else:
        pass 

   
    try:
        df = pd.read_excel(args.dataset + '_rand.xlsx', sheet_name='Sheet1')
        h_rand = df.iloc[:, 0].tolist()
        w_rand = df.iloc[:, 1].tolist()
    except FileNotFoundError:
        print(f"Warning: {args.dataset}_rand.xlsx not found. Using random crops on the fly.")
        h_rand = [0] * args.n_epochs
        w_rand = [0] * args.n_epochs

    train_list, test_list = build_datasets(
        args.root, 
        args.dataset, 
        args.image_size, 
        args.n_select_bands, 
        args.scale_ratio
    )

    
   
        
     
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
   
    criterion = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    
    model_path = f"./checkpoints/{args.dataset}_{args.arch}_{int(args.scale_ratio)}.pkl"
    
    
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

  
    
    best_psnr = 0
  
    
    for epoch in range(args.n_epochs):
        h_str = h_rand[epoch % len(h_rand)]
        w_str = w_rand[epoch % len(w_rand)]
        
     
        
  
        train(
            train_list, 
            args.image_size,
            args.scale_ratio,
            args.n_bands, 
            args.arch,
            model, 
            optimizer, 
            criterion, 
            epoch, 
            args.n_epochs,
            h_str,
            w_str
        )

       
        
        recent_psnr = validate(
            test_list, 
            args.arch,
            model, 
            epoch, 
            args.n_epochs
        )
        
   
        if recent_psnr > best_psnr:
            best_psnr = recent_psnr
            torch.save(model.state_dict(), model_path)
            print(f'New Best PSNR: {best_psnr:.4f} | Model Saved!')
        else:
            print(f'Current PSNR: {recent_psnr:.4f} | Best PSNR: {best_psnr:.4f}')

    print(f'Training Finished. Best PSNR: {best_psnr:.4f}')

if __name__ == '__main__':
    main()
