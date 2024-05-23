import wandb
import argparse
import itertools
import time

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from gan_utils import ReplayBuffer
from gan_utils import LambdaLR

from gan_utils import weights_init_normal
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=10, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/user/Gp sCT/PyTorch-CycleGAN-master/datasets/Mri_Ct', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=2, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--img_height', type=int, default=224, help='height of the images after resizing')
parser.add_argument('--img_width', type=int, default=224, help='width of the images after resizing')
#parser.add_argument('--crop_height', type=int, default=224, help='height of the random crop')
#parser.add_argument('--crop_width', type=int, default=224, help='width of the random crop')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
# parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

device = torch.device("cuda" if torch.cuda.is_available() and opt.cuda else "cpu")

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = torch.zeros(opt.batchSize, opt.input_nc, opt.img_height, opt.img_width, device='cuda')
input_B = torch.zeros(opt.batchSize, opt.output_nc, opt.img_height, opt.img_width, device='cuda')

netG_A2B.to(device)
netG_B2A.to(device)
netD_A.to(device)
netD_B.to(device)

target_real = torch.ones(opt.batchSize, 1, device=device, dtype=torch.float32)
target_fake = torch.zeros(opt.batchSize, 1, device=device, dtype=torch.float32)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_,mode='train' ,unaligned=False),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu, drop_last=True)



best_val_loss_G_A2B = float('inf')
best_val_loss_G_B2A = float('inf')
###################################
val_dataloader = DataLoader(
    ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=False, mode='val'),
    batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu, drop_last=True)


wandb.init(project="your_project_name", config={
    "learning_rate": 0.0002,
    "epochs": opt.n_epochs,
    "batch_size": opt.batchSize,
})

#################

import torch

def log_images_to_wandb(real_A, fake_B, real_B, fake_A, recovered_A, recovered_B, epoch, iteration):
    # Log the MRI -> Generated CT -> Recovered MRI cycle  in captions
    log_data = {
        f"Epoch {epoch} Iteration {iteration} - MRI to CT to Recovered MRI": [
            wandb.Image(real_A.detach().cpu() * 0.5 + 0.5, caption=f"Real MRI"),
            wandb.Image(fake_B.detach().cpu() * 0.5 + 0.5, caption="Generated CT"),
            wandb.Image(recovered_A.detach().cpu() * 0.5 + 0.5, caption="Recovered MRI")
        ],
        f"Epoch {epoch} Iteration {iteration} - CT to MRI to Recovered CT": [
            wandb.Image(real_B.detach().cpu() * 0.5 + 0.5, caption=f"Real CT"),
            wandb.Image(fake_A.detach().cpu() * 0.5 + 0.5, caption="Generated MRI"),
            wandb.Image(recovered_B.detach().cpu() * 0.5 + 0.5, caption="Recovered CT")
        ]
    }
    wandb.log(log_data)

def psnr(input, target, max_val=1.0):
    mse = torch.nn.functional.mse_loss(input, target)
    if mse == 0:
        return float('inf')  # Infinite PSNR if there is no error
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def log_differences_as_values_to_wandb(real_A, fake_B, real_B, fake_A, recovered_A, recovered_B):
    # Calculate Mean Absolute Error (MAE) for both cycles
    mae_A = torch.nn.functional.l1_loss(recovered_A, real_A)
    mae_B = torch.nn.functional.l1_loss(recovered_B, real_B)

    # Calculate Mean Squared Error (MSE) for both cycles
    mse_A = torch.nn.functional.mse_loss(recovered_A, real_A)
    mse_B = torch.nn.functional.mse_loss(recovered_B, real_B)

    # Calculate PSNR for both cycles
    psnr_A = psnr(recovered_A, real_A)
    psnr_B = psnr(recovered_B, real_B)


    # Log the errors and PSNR to W&B
    wandb.log({
        f"Metrics": {
            "MAE A to B to A": mae_A.item(),
            "MAE B to A to B": mae_B.item(),
            "MSE A to B to A": mse_A.item(),
            "MSE B to A to B": mse_B.item(),
            "PSNR A to B to A": psnr_A.item(),
            "PSNR B to A to B": psnr_B.item(),

        }
    })


def save_checkpoint(state, filename="best_checkpoint.pth.tar"):
    torch.save(state, filename)

def validate_model_separate(dataloader, netG_A2B, netG_B2A, netD_A, netD_B, criterion_GAN, criterion_cycle, criterion_identity, device):
    with torch.no_grad():
        netG_A2B.eval()
        netG_B2A.eval()
        netD_A.eval()
        netD_B.eval()
        
        total_loss_G_A2B = 0
        total_loss_G_B2A = 0
        total_cycle_loss_A2B = 0
        total_cycle_loss_B2A = 0
        total_identity_loss_A2B = 0
        total_identity_loss_B2A = 0
        
        for batch in dataloader:
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # Forward pass through generators
            fake_B = netG_A2B(real_A)
            fake_A = netG_B2A(real_B)
            recovered_A = netG_B2A(fake_B)
            recovered_B = netG_A2B(fake_A)

            # Generators' losses
            loss_GAN_A2B = criterion_GAN(netD_B(fake_B), torch.ones(fake_B.size(0), 1, device=device))
            loss_GAN_B2A = criterion_GAN(netD_A(fake_A), torch.ones(fake_A.size(0), 1, device=device))
            loss_cycle_A2B = criterion_cycle(recovered_B, real_B) * 10
            loss_cycle_B2A = criterion_cycle(recovered_A, real_A) * 10
            loss_identity_A2B = criterion_identity(fake_B, real_B) * 5
            loss_identity_B2A = criterion_identity(fake_A, real_A) * 5

            # Accumulate losses for each generator
            total_loss_G_A2B += (loss_GAN_A2B + loss_cycle_A2B + loss_identity_A2B).item()
            total_loss_G_B2A += (loss_GAN_B2A + loss_cycle_B2A + loss_identity_B2A).item()
            total_cycle_loss_A2B += loss_cycle_A2B.item()
            total_cycle_loss_B2A += loss_cycle_B2A.item()
            total_identity_loss_A2B += loss_identity_A2B.item()
            total_identity_loss_B2A += loss_identity_B2A.item()

        # Calculate average losses
        avg_loss_G_A2B = total_loss_G_A2B / len(dataloader)
        avg_loss_G_B2A = total_loss_G_B2A / len(dataloader)
        avg_cycle_loss_A2B = total_cycle_loss_A2B / len(dataloader)
        avg_cycle_loss_B2A = total_cycle_loss_B2A / len(dataloader)
        avg_identity_loss_A2B = total_identity_loss_A2B / len(dataloader)
        avg_identity_loss_B2A = total_identity_loss_B2A / len(dataloader)

        # Log validation losses separately
        wandb.log({
            'val_loss_G_A2B': avg_loss_G_A2B,
            'val_cycle_loss_A2B': avg_cycle_loss_A2B,
            'val_identity_loss_A2B': avg_identity_loss_A2B,
            'val_loss_G_B2A': avg_loss_G_B2A,
            'val_cycle_loss_B2A': avg_cycle_loss_B2A,
            'val_identity_loss_B2A': avg_identity_loss_B2A
        })

        return avg_loss_G_A2B, avg_loss_G_B2A


                    # Initial setup
start_time = time.time()
save_interval = 60 * 60  # 30 minutes, for example

iteration_count = 0  
checkpoint_interval = 50

import torch.nn.functional as F

#def segmentation_loss(segmented_real, segmented_fake):
#    return F.mse_loss(segmented_real, segmented_fake)


def segmentation_loss(segmented_real, segmented_fake, smooth=1.0):
    # Ensure tensors have the correct dimensions
    if segmented_real.dim() != 3 or segmented_fake.dim() != 3:
        raise ValueError("Input tensors must be 3-dimensional (batch_size, height, width)")
    
    intersection = (segmented_real * segmented_fake).sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (segmented_real.sum(dim=(1, 2)) + segmented_fake.sum(dim=(1, 2)) + smooth)
    return 1 - dice.mean()


from inference import segment_image

from torchvision.transforms import ToPILImage

def tensor_to_image_file(tensor, filepath):
    # Check if the tensor has a batch dimension and select the first image
    if tensor.dim() == 4:
        tensor = tensor[0]  # This selects the first image from the batch
    # Convert to CPU and detach from gradients
    tensor = tensor.cpu().detach()
    # Convert to PIL Image
    transform = ToPILImage()
    image = transform(tensor)
    # Save the image to the filepath
    image.save(filepath)

from torchvision.io import read_image
from torchvision.transforms import ConvertImageDtype

def load_segmented_image_as_tensor(filepath):
    # Load the image file, ensuring it is converted to a tensor
    image = read_image(filepath)
    # Normalize and convert the image to the expected dtype
    image = ConvertImageDtype(torch.float32)(image) / 255
    return image

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()
    for i, batch in enumerate(dataloader):
        #########################################
        #########################################
        # Set model input
        #real_A = Variable(input_A.copy_(batch['A'])).to(device)
        #real_B = Variable(input_B.copy_(batch['B'])).to(device)

        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)


        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)


       
        

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10 #check if 10 can be changed

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10


        # Assume 'segment_image' function is adapted to return a tensor of the segmented image
        # Create file path
        temp_A_path = "temp_A_image.png"
        temp_recovered_A_path = "temp_r_A_image.png"

        tensor_to_image_file(real_A, temp_A_path)
        tensor_to_image_file(recovered_A, temp_recovered_A_path)



        segment_image(temp_A_path, "/home/user/Gp sCT/PyTorch-CycleGAN-master/predictions_dir", "/home/user/Gp sCT/PyTorch-CycleGAN-master/unet.pt")
    
        segment_image(temp_recovered_A_path, "/home/user/Gp sCT/PyTorch-CycleGAN-master/predictions_dir", "/home/user/Gp sCT/PyTorch-CycleGAN-master/unet.pt")
        
        # Before calling segmentation_loss
        segmented_real_A = load_segmented_image_as_tensor('/home/user/Gp sCT/PyTorch-CycleGAN-master/predictions_dir/temp_A_image_segmented.png')
        segmented_recovered_A = load_segmented_image_as_tensor('/home/user/Gp sCT/PyTorch-CycleGAN-master/predictions_dir/temp_r_A_image_segmented.png')



        if segmented_real_A is None or segmented_recovered_A is None:
            print("Segmentation failed, skipping loss calculation for this batch.")
        else:
            seg_loss_A = segmentation_loss(segmented_real_A, segmented_recovered_A)

        


        temp_B_path = "temp_B_image.png"
        temp_recovered_B_path = "temp_r_B_image.png"

        tensor_to_image_file(real_B, temp_B_path)
        tensor_to_image_file(recovered_B, temp_recovered_B_path)



        segment_image(temp_B_path, "/home/user/Gp sCT/PyTorch-CycleGAN-master/predictions_dir", "/home/user/Gp sCT/PyTorch-CycleGAN-master/unet.pt")
    
        segment_image(temp_recovered_B_path, "/home/user/Gp sCT/PyTorch-CycleGAN-master/predictions_dir", "/home/user/Gp sCT/PyTorch-CycleGAN-master/unet.pt")
        
        # Before calling segmentation_loss
        segmented_real_B = load_segmented_image_as_tensor('/home/user/Gp sCT/PyTorch-CycleGAN-master/predictions_dir/temp_A_image_segmented.png')
        segmented_recovered_B = load_segmented_image_as_tensor('/home/user/Gp sCT/PyTorch-CycleGAN-master/predictions_dir/temp_r_A_image_segmented.png')

        if segmented_real_B is None or segmented_recovered_B is None:
            print("Segmentation failed, skipping loss calculation for this batch.")
        else:
            seg_loss_B = segmentation_loss(segmented_real_B, segmented_recovered_B)


          
        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + seg_loss_A + seg_loss_B
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################
     

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()


        optimizer_D_B.step()

        log_differences_as_values_to_wandb(
        real_A[0], fake_B[0], real_B[0], fake_A[0], recovered_A[0], recovered_B[0]
    )

        if i % 10 == 0:  # Log every 10 iterations
            log_images_to_wandb(
        real_A[0], fake_B[0], real_B[0], fake_A[0], recovered_A[0], recovered_B[0], epoch, i
    )
    
   
     ##############################################################
        

        iteration_count += 1  # Increment iteration count
        print(f"Iteration: {iteration_count}")
        # Check if it's time to save a checkpoint

        if iteration_count >= checkpoint_interval:
            print(f"Saving checkpoint at epoch {epoch}, iteration {i}")
            save_checkpoint({
                'epoch': epoch,
                'iteration': i,
                'state_dict_G_A2B': netG_A2B.state_dict(),
                'state_dict_G_B2A': netG_B2A.state_dict(),
                'state_dict_D_A': netD_A.state_dict(),
                'state_dict_D_B': netD_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_A': optimizer_D_A.state_dict(),
                'optimizer_D_B': optimizer_D_B.state_dict(),
            }, filename=f"iteration_checkpoint_epoch_{epoch}.pth.tar")
            iteration_count = 0  # Reset iteration count for the next checkpoint
    
        ##############################################################
        ###################################
        print(f'Epoch: {epoch}/{opt.n_epochs}', {'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                     'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)})

        if (epoch >1) and i % 50 == 0:  # Validate every 50 iterations on every 3rd epoch
            device = torch.device("cuda" if torch.cuda.is_available() and opt.cuda else "cpu")
            print("Performing validation...")
            val_loss_G_A2B, val_loss_G_B2A = validate_model_separate(val_dataloader, netG_A2B, netG_B2A, netD_A, netD_B, criterion_GAN, criterion_cycle, criterion_identity, device)
            print(f'Validation Loss at Epoch {epoch + 1}, Iteration {i}: {val_loss_G_A2B},{val_loss_G_B2A}')
            if val_loss_G_A2B < best_val_loss_G_A2B:
                best_val_loss_G_A2B = val_loss_G_A2B
                torch.save(netG_A2B.state_dict(), 'seg_best_netG_A2B.pth')
                print(f"New best model for A2B saved with validation loss {best_val_loss_G_A2B}")

            if val_loss_G_B2A < best_val_loss_G_B2A:
                best_val_loss_G_B2A = val_loss_G_B2A
                torch.save(netG_B2A.state_dict(), 'seg_best_netG_B2A.pth')
                print(f"New best model for B2A saved with validation loss {best_val_loss_G_B2A}")

            # Additionally, save a regular checkpoint every 3 epochs
            save_checkpoint({
                'epoch': epoch,
                'state_dict_G_A2B': netG_A2B.state_dict(),
                'state_dict_G_B2A': netG_B2A.state_dict(),
                'state_dict_D_A': netD_A.state_dict(),
                'state_dict_D_B': netD_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_A': optimizer_D_A.state_dict(),
                'optimizer_D_B': optimizer_D_B.state_dict(),
            }, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")


        # Log losses and images
        wandb.log({
            'loss_G': loss_G.item(),
            'loss_G_identity': (loss_identity_A + loss_identity_B).item(),
            'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A).item(),
            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB).item(),
            'loss_D_A': loss_D_A.item(),
            'loss_D_B': loss_D_B.item(),
            'epoch': epoch,
        
            })

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()


    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')
   
###################################
wandb.finish()