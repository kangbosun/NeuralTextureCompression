

import os
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler as profiler
from torchvision.models import vgg16
from torch.nn.functional import mse_loss
import torchvision.transforms.functional as TF
import numpy as np
import tqdm
import piqa

class EncoderSettings:
    def __init__(self, input_size, featuregrids_num=2):
        self.input_channels = input_size
        self.channels_per_featuregrid = 3
        self.featuregrids_num = featuregrids_num
        self.layers = [64, 64]  # 2 for positional encoding

    def getEncoderOutputChannels(self):
        return self.channels_per_featuregrid * self.featuregrids_num
    
    def getDecoderInputChannels(self):
        return self.getEncoderOutputChannels() + 2  # 2 for positional encoding

class AutoEncoder(nn.Module):
    def __init__(self, encoderSettings):
        super(AutoEncoder, self).__init__()

        self.encoderSettings = encoderSettings
        
        layers = encoderSettings.layers
        self.encoder = nn.Sequential(
            nn.Conv2d(encoderSettings.input_channels + 2, layers[0], 1),
            nn.ReLU(True),
            nn.Conv2d(layers[0], layers[1], 1),
            nn.ReLU(True),
            nn.Conv2d(layers[1], encoderSettings.getEncoderOutputChannels(), 1),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(encoderSettings.getDecoderInputChannels(), layers[1], 1),
            nn.ReLU(True),
            nn.Conv2d(layers[1], layers[0], 1),
            nn.ReLU(True),
            nn.Conv2d(layers[0], encoderSettings.input_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)  # 압축
        output = self.decoder(z)  # 복원
        return output


def positional_encoding_xy(x, y, num_frequencies):
    return [y, x]

    """
    Generate positional encodings for 2D coordinates (x, y).

    Args:
        x (float or np.ndarray): The x-coordinate(s).
        y (float or np.ndarray): The y-coordinate(s).
        num_frequencies (int): Number of frequency components for encoding.

    Returns:
        np.ndarray: The positional encoding of shape (..., 4 * num_frequencies).
    """
    if isinstance(x, (float, int)):
        x = np.array([x])
    if isinstance(y, (float, int)):
        y = np.array([y])

    x, y = np.atleast_1d(x), np.atleast_1d(y)

    # Frequencies for positional encoding
    frequencies = 2 ** np.arange(num_frequencies)

    # Compute sine and cosine components for x and y
    sin_x = np.sin(x[:, None] * frequencies)
    cos_x = np.cos(x[:, None] * frequencies)
    sin_y = np.sin(y[:, None] * frequencies)
    cos_y = np.cos(y[:, None] * frequencies)

    # Concatenate results
    encoding = np.concatenate([sin_x, cos_x, sin_y, cos_y], axis=-1).astype(np.float32).ravel()

    return encoding

import numpy as np

# Custom dataset class for patch_size
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, patch_size):
        self.dataset = torch.Tensor(dataset).permute(2, 0, 1)
        self.C, self.H, self.W = self.dataset.shape
        self.patch_size = patch_size
        self.num_patches = int(self.H * self.W / (self.patch_size * self.patch_size))

        print("===== PatchDataset Info =====")
        print("Dataset shape: ", self.dataset.shape)
        print("Patch size: ", self.patch_size)
        print("Number of patches: ", self.num_patches)
        print("==============================")

        pos = np.zeros((2, self.H, self.W))

        pos[0,:,:] = np.arange(self.W)[None, :] / self.W
        pos[1,:,:] = np.arange(self.H)[:, None] / self.H
        
        self.pos = torch.Tensor(pos)

    def __len__(self):
        return self.num_patches
    
    def __getitem__(self, idx):
        y = int(idx // (self.W / self.patch_size)) * self.patch_size
        x = int(idx % (self.W / self.patch_size)) * self.patch_size

        patch = self.dataset[:, y:y+self.patch_size, x:x+self.patch_size]

        pos = self.pos[:, y:y+self.patch_size, x:x+self.patch_size]

        return patch, pos

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, nSamples=None):
 
        print("Initializing CustomDataset")

        print("Creating dataset tensor")
        self.dataset = torch.Tensor(dataset)
        self.height, self.width = dataset.shape[:2]
        self.nSamples = nSamples if nSamples is not None else self.width * self.height

        print("Creating position tensor")
        progress_bar = tqdm.tqdm(total=self.height, desc="Creating position tensor")
        pos_array = []
        for y in range(self.height):
            for x in range(self.width):
                pos_array.append([x / self.width, y / self.height])

            progress_bar.update(1)
        progress_bar.close()
        self.pos_tensor = torch.Tensor(pos_array)

    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, idx):
        x = idx % self.width
        y = idx // self.width

        tex_tensor = self.dataset[y, x]
        return tex_tensor, self.pos_tensor[y * self.width + x]


class SSIMLoss(nn.Module):
    def __init__(self, patch_size=4):
        super(SSIMLoss, self).__init__()
        self.ssim = piqa.SSIM(window_size=patch_size, n_channels=8).to('cuda')
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        #check shape
        #print("Output shape: ", output.shape)
        #print("Target shape: ", target.shape)

        #output : N x W x H x C
        # convert to rgb first 3 channels
        color_output = output
        color_target = target

        ssim = self.ssim(color_output, color_target)
        ssim_mean = torch.mean(ssim)

        ssim_loss = 1 - ssim_mean

        #combine mse and ssim
        
        return ssim_loss


def plot_loss_psnr(losses, psnrs, ssims, outputDirectory):
    """
    Loss 및 PSNR 그래프를 업데이트하는 함수
    """
    plt.figure(figsize=(10, 10))
    
    # Loss 그래프
    plt.subplot(2, 2, 1)
    plt.plot(losses, label="Loss", color='red', marker='o', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    # PSNR 그래프
    plt.subplot(2, 2, 2)
    plt.plot(psnrs, label="PSNR", color='blue', marker='o', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Over Epochs")
    plt.legend()

    # SSIM 그래프
    plt.subplot(2, 2, 3)
    plt.plot(ssims, label="SSIM", color='green', marker='o', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM Over Epochs")
    plt.legend()

    # 그래프 저장
    save_path = os.path.join(outputDirectory, "training_curve.png")
    plt.savefig(save_path)
    plt.close()  # 그래프가 계속 쌓이는 것을 방지

def end_of_epoch(autoencoder, outputDirectory, epoch, loss_history, psnr_history, ssim_history, lr):
    """
    한 epoch이 끝날 때마다 호출되는 함수
    """
    #기존 모델 삭제
    model_path = os.path.join(outputDirectory, f"model.pth")
    if os.path.exists(model_path):
        os.remove(model_path)

    # 모델 저장
    model_path = os.path.join(outputDirectory, f"model.pth")
    torch.save(autoencoder.state_dict(), model_path)

    # Loss 및 PSNR 그래프 업데이트
    plot_loss_psnr(loss_history, psnr_history, ssim_history, outputDirectory)

    # Loss 및 PSNR CSV에 저장
    csv_path = os.path.join(outputDirectory, "training_curve.csv")
    if epoch == 0:
        with open(csv_path, "w") as f:
            f.write("Epoch,Loss,PSNR,SSIM,LearningRate\n")
    with open(csv_path, "a") as f:
        f.write(f"{epoch},{loss_history[-1]},{psnr_history[-1]},{ssim_history[-1]},{lr[0]}\n")

def trainAutoEncoder(autoencoder, device, dataset, epochs, batchSize, learningRate, outputDirectory):
    #load previous model and log
    start_epoch = 0
    loss_history = []
    psnr_history = []
    ssim_history = []
    start_lr = learningRate
    
    #load loss and psnr history
    last_model = outputDirectory + 'model.pth'

    if os.path.exists(outputDirectory + 'training_curve.csv') and os.path.exists(outputDirectory + 'model.pth'):       
        autoencoder.load_state_dict(torch.load(last_model))
        print("Loaded model: ", last_model)  

        with open(outputDirectory + 'training_curve.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.split(',')
                start_epoch = int(parts[0]) + 1
                loss_history.append(float(parts[1]))
                psnr_history.append(float(parts[2]))
                ssim_history.append(float(parts[3]))
                start_lr = float(parts[4])

            #print last epoch info
            print("Last epoch: ", start_epoch, " Loss: ", loss_history[-1], " PSNR: ", psnr_history[-1])          

    autoencoder.to(device)
    autoencoder.train()
    
    learningRate = start_lr
    optimizer = optim.Adam(autoencoder.parameters(), lr=learningRate)
    
    from torch.optim.lr_scheduler import StepLR
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    mse_loss_func = nn.MSELoss()
    ssim_loss_func = SSIMLoss()

    #data info
    patch_size = 8
    batch_size = patch_size * patch_size
    batch_num = (int)(batchSize / batch_size)
    patch_dataset = PatchDataset(dataset, patch_size)
  
    data_loader = torch.utils.data.DataLoader(patch_dataset, batch_size=batch_num, shuffle=True, num_workers=0)
    

    print("===== Dataset Info =====")
    print("Dataset size: ", len(dataset))
    print("Batch size: ", patch_size, "x", patch_size, 'x', batch_num, '=', batchSize)
    print("========================")

    #batch_dataset = CustomDataset(dataset)
    #data_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=batchSize, shuffle=True, num_workers=0)

    num_feature_grids = autoencoder.encoderSettings.featuregrids_num

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoce_mse_loss = 0.0
        epoch_PSNR  = 0
        epoch_SSIM = 0

        batch_bar = tqdm.tqdm(total=len(data_loader), desc="Epoch " + str(epoch + 1) + "/" + str(epochs), leave=True)
        for texTensor, posTensor in data_loader:
            texTensor = texTensor.to(device)
            posTensor = posTensor.to(device)

            optimizer.zero_grad()

            encoder_inputs = torch.cat([texTensor, posTensor], dim=1)
            
            endcoder_outputs = autoencoder.encoder(encoder_inputs)
 
            fine_outputs = endcoder_outputs[:, :3]

            decoder_inputs = fine_outputs

            #create feature grid inputs
            scale_factor = 0.5
            for i in range(1, num_feature_grids):
                coarse_outputs = endcoder_outputs[:, 3 * i:3 * (i + 1), :, :]
                coarse_downscaled_outputs = nn.functional.interpolate(coarse_outputs, scale_factor=scale_factor, mode='bilinear', align_corners=False)
                coarse_upscaled_outputs = nn.functional.interpolate(coarse_downscaled_outputs, size=(fine_outputs.shape[2], fine_outputs.shape[3]), mode='bilinear', align_corners=False)
                decoder_inputs = torch.cat([decoder_inputs, coarse_upscaled_outputs], dim=1)
                scale_factor *= 0.5

            decoder_inputs = torch.cat([decoder_inputs, posTensor], dim=1)

            #decoder output
            decoder_outputs = autoencoder.decoder(decoder_inputs)

            mse_loss = mse_loss_func(decoder_outputs, texTensor)
            ssim_loss = ssim_loss_func(decoder_outputs, texTensor)

            loss = mse_loss# + ssim_loss
            loss.backward()

            epoch_PSNR += 10 * torch.log10(1 / mse_loss)
            epoch_loss += loss.item()
            epoch_SSIM += 1 - ssim_loss.item()
            epoce_mse_loss += mse_loss.item()

            optimizer.step()
            
            batch_bar.update(1)

        batch_bar.close()

        epoch_loss = epoch_loss / len(data_loader)
        epoch_PSNR = epoch_PSNR / len(data_loader)
        epoch_SSIM = epoch_SSIM / len(data_loader)
        epoce_mse_loss = epoce_mse_loss / len(data_loader)

        loss_history.append(epoch_loss)
        psnr_history.append(epoch_PSNR.cpu().detach().numpy()) 
        ssim_history.append(epoch_SSIM)

        print(f"Epoch {epoch + 1}/{epochs} Loss: {epoch_loss:.6f} MSE: {epoce_mse_loss:.6f} PSNR: {epoch_PSNR:.4f} SSIM: {epoch_SSIM:.4f} LR: {scheduler.get_last_lr()}")
    
        end_of_epoch(autoencoder, outputDirectory, epoch, loss_history, psnr_history, ssim_history, scheduler.get_last_lr())
                                                        
        scheduler.step(epoch_loss) 
                                                                
 