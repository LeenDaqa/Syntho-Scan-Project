import argparse
import os
import sys
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torchvision.transforms import Resize
from models import Generator
from datasets import ImageDataset
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse


import wandb


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/Mri_Ct', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=224, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='/home/user/Gp sCT/PyTorch-CycleGAN-master/seg_best_netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='/home/user/Gp sCT/PyTorch-CycleGAN-master/seg_best_netG_B2A.pth', help='B2A generator checkpoint file')
#parser.add_argument('--checkpoint', type=str, default='/home/user/Gp sCT/PyTorch-CycleGAN-master/checkpoint_epoch_7.pth.tar', help='path to the checkpoint file')
opt = parser.parse_args()
print(opt)

# Networks initialization
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load the checkpoint
#checkpoint = torch.load(opt.checkpoint)

# Load state dicts for generators using the correct keys from the checkpoint
#netG_A2B.load_state_dict(checkpoint['state_dict_G_A2B'])
#netG_B2A.load_state_dict(checkpoint['state_dict_G_B2A'])
netG_A2B.eval()
netG_B2A.eval()


# Load weights based on file name
netG_A2B.load_state_dict(torch.load('seg_best_netG_A2B.pth'))
netG_B2A.load_state_dict(torch.load('seg_best_netG_B2A.pth'))
# Set model's test mode



# Inputs & targets memory allocation
device = torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')
input_A = torch.zeros(opt.batchSize, opt.input_nc, opt.size, opt.size, device=device, dtype=torch.float)
input_B = torch.zeros(opt.batchSize, opt.output_nc, opt.size, opt.size, device=device, dtype=torch.float)

# Dataset loader
transforms_ = [Resize((opt.size, opt.size)),
               transforms.ToTensor(),
               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
               
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test', unaligned=False),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('/home/user/Gp sCT/PyTorch-CycleGAN-master/output/A'):
    os.makedirs('/home/user/Gp sCT/PyTorch-CycleGAN-master/output/A')
if not os.path.exists('/home/user/Gp sCT/PyTorch-CycleGAN-master/output/B'):
    os.makedirs('/home/user/Gp sCT/PyTorch-CycleGAN-master/output/B')


# Before the loop
psnr_A_values = []
mse_A_values = []
psnr_B_values = []
mse_B_values = []

wandb.init(project='CycleGAN_Testing', entity='ham20190047', name='Mri_Ct_Test')



with open('best_metrics_log.txt', 'w') as file:
    # Metrics storage lists
    psnr_values = []
    mse_values = []

    for i, batch in enumerate(dataloader):
        # Set model input
        # Instead of reusing input_A and input_B
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        with torch.no_grad():
            fake_B = netG_A2B(real_A)
            fake_A = netG_B2A(real_B)

            # Convert tensors to numpy arrays
        real_A_np = real_A.cpu().detach().numpy() * 0.5 + 0.5
        real_B_np = real_B.cpu().detach().numpy() * 0.5 + 0.5
        fake_A_np = fake_A.cpu().detach().numpy() * 0.5 + 0.5
        fake_B_np = fake_B.cpu().detach().numpy() * 0.5 + 0.5
                # Log images
        wandb.log({
        'Real A -> Fake B': [wandb.Image(torch.cat((real_A.detach().cpu()* 0.5 + 0.5, fake_B.detach().cpu()* 0.5 + 0.5), dim=3), caption="Real A -> Fake B")],
        'Real B -> Fake A': [wandb.Image(torch.cat((real_B.detach().cpu()* 0.5 + 0.5, fake_A.detach().cpu()* 0.5 + 0.5), dim=3), caption="Real B -> Fake A")]
        })


# Calculate metrics
        psnr_AB = compare_psnr(real_B_np, fake_A_np, data_range=real_B_np.max() - real_B_np.min())
        mse_AB = compare_mse(real_B_np, fake_A_np)

        psnr_BA = compare_psnr(real_A_np, fake_B_np, data_range=real_A_np.max() - real_A_np.min())
        mse_BA = compare_mse(real_A_np, fake_B_np)




            # Log metrics
        wandb.log({'PSNR AB': psnr_AB, 'MSE AB': mse_AB, 'PSNR BA': psnr_BA, 'MSE BA': mse_BA})




        # Inside the loop, after metrics calculation
        psnr_A_values.append(psnr_AB)
        mse_A_values.append(mse_AB)
        psnr_B_values.append(psnr_BA)
        mse_B_values.append(mse_BA)
        # Write metrics to file for each image set
        file.write(f"Image {i+1} Metrics for Fake A vs Real B: PSNR: {psnr_AB:.2f}, MSE: {mse_AB:.2f}\n")
        file.write(f"Image {i+1} Metrics for Fake B vs Real A: PSNR: {psnr_BA:.2f}, MSE: {mse_BA:.2f}\n")



        if not fake_A.any() or not fake_B.any():
            print("Generated tensors are empty")
        print(fake_A.shape)
        print(fake_B.shape)

        # Save image files
        save_image(fake_A, '/home/user/Gp sCT/PyTorch-CycleGAN-master/output/A/%04d.png' % (i+1))
        save_image(fake_B, '/home/user/Gp sCT/PyTorch-CycleGAN-master/output/B/%04d.png' % (i+1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    # Calculate average metrics and write to file
    file.write(f"\nAverage Metrics for Fake A: PSNR: {np.mean(psnr_A_values):.2f}, MSE: {np.mean(mse_A_values):.2f}\n")
    file.write(f"Average Metrics for Fake B: PSNR: {np.mean(psnr_B_values):.2f}, MSE: {np.mean(mse_B_values):.2f}\n")

sys.stdout.write('\n')
