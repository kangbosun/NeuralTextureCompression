

import os
import random
import cv2
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
import json

class EncoderSettings:
    def __init__(self, input_size, featuregrids_num=2):
        self.input_channels = input_size
        self.channels_per_featuregrid = 3
        self.featuregrids_num = featuregrids_num
        self.layers = [32]  # 2 for positional encoding

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
            #nn.Conv2d(layers[0], layers[1], 1),
            #nn.ReLU(True),
            nn.Conv2d(layers[0], encoderSettings.getEncoderOutputChannels(), 1),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(encoderSettings.getDecoderInputChannels(), layers[0], 1),
            nn.ReLU(True),
            #nn.Conv2d(layers[1], layers[0], 1),
            #nn.ReLU(True),
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

class RandomSamplingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = torch.Tensor(dataset).permute(2, 0, 1)
        self.C, self.H, self.W = self.dataset.shape

    def __len__(self):
        return self.H * self.W
    
    def __getitem__(self, idx):
        y = int(idx // self.W)
        x = int(idx % self.W)

        uv_y = y / self.H + (random.randomrange(-1, 1) / self.H)
        uv_x = x / self.W + (random.randomrange(-1, 1) / self.W)

        pos = torch.Tensor([uv_y, uv_x])

        tex = torch.nn.functional.grid_sample(self.dataset, pos, mode='bilinear', align_corners=False)

        item = torch.cat([tex, pos], dim=0)

        return item


class SSIMLoss(nn.Module):
    def __init__(self, channels, patch_size=4):
        super(SSIMLoss, self).__init__()
        self.ssim = piqa.SSIM(window_size=patch_size, n_channels=channels).to('cuda')
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


def plot_loss_psnr(losses, psnrs, outputDirectory):
    """
    Loss 및 PSNR 그래프를 업데이트하는 함수
    """
    plt.figure(figsize=(10, 10))
    
    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Loss", color='red', marker='o', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    # PSNR 그래프
    plt.subplot(1, 2, 2)
    plt.plot(psnrs, label="PSNR", color='blue', marker='o', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Over Epochs")
    plt.legend()

    # 그래프 저장
    save_path = os.path.join(outputDirectory, "training_curve.png")
    plt.savefig(save_path)
    plt.close()  # 그래프가 계속 쌓이는 것을 방지

def end_of_epoch(autoencoder, outputDirectory, epoch, loss_history, psnr_history, lr):
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
    plot_loss_psnr(loss_history, psnr_history, outputDirectory)

    # Loss 및 PSNR CSV에 저장
    csv_path = os.path.join(outputDirectory, "training_curve.csv")
    if epoch == 0:
        with open(csv_path, "w") as f:
            f.write("Epoch,Loss,PSNR,LearningRate\n")
    with open(csv_path, "a") as f:
        f.write(f"{epoch},{loss_history[-1]},{psnr_history[-1]},{lr[0]}\n")

def trainAutoEncoder(autoencoder, device, dataset, channels_list, epochs, batchSize, learningRate, outputDirectory):
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

    total_channels = 0
    for channels in channels_list:
        total_channels += channels

    mse_loss_func = nn.MSELoss()
    ssim_loss_func = SSIMLoss(total_channels)

    data_set_size = len(dataset[0])
    #trains using full res feature grid
    top_level_grid_size = 2048
    top_level_grid_ratio = top_level_grid_size / data_set_size
    num_feature_grids = autoencoder.encoderSettings.featuregrids_num

    top_level_feature_grid_patch_size = int(2 ** (num_feature_grids ))

    #data info
    num_samples = 10000
    patch_size = int(top_level_feature_grid_patch_size / top_level_grid_ratio)
    batch_size = patch_size * patch_size
    batch_num = max(1, (int)(batchSize / batch_size))
    patch_dataset = PatchDataset(dataset, patch_size)

    #random_dataset = RandomSamplingDataset(dataset)
  
    data_loader = torch.utils.data.DataLoader(patch_dataset, batch_size=batch_num, shuffle=True, num_workers=0)
    

    print("===== Dataset Info =====")
    print("Dataset size: ", len(dataset))
    print("Batch size: ", patch_size, "x", patch_size, 'x', batch_num, '=', batchSize)
    print("========================")

    #batch_dataset = CustomDataset(dataset)
    #data_loader = torch.utils.data.DataLoader(batch_dataset, batch_size=batchSize, shuffle=True, num_workers=0)

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

            feature_grid_size = top_level_feature_grid_patch_size
            for i in range(0, num_feature_grids):
                grid_output = endcoder_outputs[:, 3 * i:3 * (i + 1), :, :]
                if feature_grid_size != top_level_feature_grid_patch_size:
                    grid_output = nn.functional.interpolate(grid_output, size=(feature_grid_size, feature_grid_size), mode='nearest')
                    grid_output = nn.functional.interpolate(grid_output, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                
                if i == 0:
                    decoder_inputs = grid_output
                else:              
                    decoder_inputs = torch.cat([decoder_inputs, grid_output], dim=1)

                feature_grid_size = int(feature_grid_size / 2)

            decoder_inputs = torch.cat([decoder_inputs, posTensor], dim=1)

            #decoder output
            decoder_outputs = autoencoder.decoder(decoder_inputs)

            #get loss for each channels
            mse_loss = mse_loss_func(decoder_outputs, texTensor)
            #ssim_loss = ssim_loss_func(decoder_outputs, texTensor)

            loss = mse_loss# + ssim_loss
            loss.backward()

            epoch_PSNR += 10 * torch.log10(1 / mse_loss)
            epoch_loss += loss.item()
            epoch_SSIM += 0#1 - ssim_loss.item()
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


class PatchDataset2(torch.utils.data.Dataset):
    def __init__(self, device, tex_tensor, y_tensor, x_tensor, patch_size):
        self.tex_tensor = tex_tensor
        self.y_tensor = y_tensor
        self.x_tensor = x_tensor
        self.pos_tensor = torch.stack([x_tensor, y_tensor], dim=-1)
        self.pos_tensor = self.pos_tensor * 2 - 1
        self.patch_size = patch_size
        self.C, self.H, self.W = tex_tensor.shape
        self.num_patches = int(self.H * self.W / (self.patch_size * self.patch_size))
        self.data_set_size = self.H * self.patch_size
        self.device = device

        print("===== PatchDataset Info =====")
        print("Dataset shape: ", self.tex_tensor.shape)
        print("Patch size: ", self.patch_size)
        print("Number of patches: ", self.num_patches)
        print("==============================")

    def __len__(self):
        return self.num_patches
    
    def __getitem__(self, idx):
        y = int(idx // (self.W / self.patch_size)) * self.patch_size
        x = int(idx % (self.W / self.patch_size)) * self.patch_size

        patch = self.tex_tensor[:, y:y+self.patch_size, x:x+self.patch_size]
        y_tensor = self.y_tensor[y:y+self.patch_size, x:x+self.patch_size]
        x_tensor = self.x_tensor[y:y+self.patch_size, x:x+self.patch_size]
        grid = torch.stack([x_tensor, y_tensor], dim=-1)

        return patch, grid, y_tensor, x_tensor

def trainAutoEncoder2(autoencoder, device, dataset, channels_list, epochs, batchSize, learningRate, outputDirectory):
    #load previous model and log
    start_epoch = 0
    loss_history = []
    psnr_history = []

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
                start_lr = float(parts[3])

            #print last epoch info
            print("Last epoch: ", start_epoch, " Loss: ", loss_history[-1], " PSNR: ", psnr_history[-1])          

    autoencoder.to(device)
    autoencoder.train()
    
    learningRate = start_lr
    optimizer = optim.Adam(autoencoder.parameters(), lr=learningRate)
    
    from torch.optim.lr_scheduler import StepLR
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    total_channels = 0
    for channels in channels_list:
        total_channels += channels

    mse_loss_func = nn.MSELoss()

    data_set_size = len(dataset[0])
    #trains using full res feature grid
    top_level_grid_size = 1024
    top_level_grid_ratio = top_level_grid_size / data_set_size
    num_feature_grids = autoencoder.encoderSettings.featuregrids_num

    #data info
    print("===== Dataset Info =====")
    print("Dataset size: ", len(dataset))
    print("Batch size: ", batchSize)
    print("========================")

    # meshgrid
    y, x = np.meshgrid(np.linspace(0, 1, data_set_size), np.linspace(0, 1, data_set_size), indexing='ij')

    # reaarange to -1, 1
    x = x * 2 - 1
    y = y * 2 - 1

    y_tensor = torch.Tensor(y).to(device)
    x_tensor = torch.Tensor(x).to(device)

    tex_tensor = torch.Tensor(dataset).permute(2, 0, 1).to(device)

    #add batch dimension temporarily
    tex_tensor = tex_tensor.unsqueeze(0)

    # grid sample
    grid = torch.stack([x_tensor, y_tensor], dim=-1).unsqueeze(0)

    #tex_tensor = torch.nn.functional.grid_sample(tex_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    #remove batch dimension
    tex_tensor = tex_tensor.squeeze(0)
    y_tensor = y_tensor.squeeze(0)
    x_tensor = x_tensor.squeeze(0)

    patch_size = 32
    feature_grid_divider = data_set_size / patch_size
    dataloader = torch.utils.data.DataLoader(PatchDataset2(device, tex_tensor, y_tensor, x_tensor, patch_size), batch_size=1, shuffle=False, num_workers=0)

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoce_mse_loss = 0.0
        epoch_PSNR  = 0

        progress_bar = tqdm.tqdm(total=len(dataloader), desc="Epoch " + str(epoch + 1) + "/" + str(epochs), leave=True)
        for tex_patch_tensor, grid_tensor, y_tensor, x_tensor in dataloader:

            optimizer.zero_grad()

            unsigned_pos_tensor = torch.stack([y_tensor, x_tensor], dim=-1).permute(0, 3, 1, 2)
            encoder_inputs = torch.cat([tex_patch_tensor, unsigned_pos_tensor], dim=1)
            
            endcoder_outputs = autoencoder.encoder(encoder_inputs)
            #decoder_inputs = endcoder_outputs
            decoder_inputs = None

            feature_grid_size = (int)(top_level_grid_size / feature_grid_divider)
            for i in range(0, num_feature_grids):
                grid_output = endcoder_outputs[:, 3 * i:3 * (i + 1), :, :]
                if feature_grid_size != data_set_size:
                    grid_output = nn.functional.interpolate(grid_output, size=(feature_grid_size, feature_grid_size), mode='bilinear', align_corners=True)
                    grid_output = nn.functional.interpolate(grid_output, size=(patch_size, patch_size), mode='bilinear', align_corners=True)
                    #grid_output = torch.nn.functional.grid_sample(grid_output, pos_tensor, mode='bilinear', align_corners=True)
                    
                    decoder_inputs = grid_output if decoder_inputs is None else torch.cat([decoder_inputs, grid_output], dim=1)
            
                feature_grid_size = min(256, int(feature_grid_size / 2))

            decoder_inputs = torch.cat([decoder_inputs, unsigned_pos_tensor], dim=1)

            #decoder output
            decoder_outputs = autoencoder.decoder(decoder_inputs)

            # tex_image = decoder_outputs[0, 0:3].permute(1, 2, 0).cpu().detach().numpy()
            # tex_image = tex_image * 255
            # cv2.imwrite(outputDirectory + 'sample_tex.png', tex_image)            

            #get loss for each channels
            mse_loss = mse_loss_func(decoder_outputs, tex_patch_tensor)

            loss = mse_loss
            loss.backward()

            epoch_PSNR += 10 * torch.log10(1 / mse_loss)
            epoch_loss += loss.item()
            epoce_mse_loss += mse_loss.item()

            optimizer.step()

            progress_bar.update(1)

        progress_bar.close()

        epoch_loss = epoch_loss / len(dataloader)
        epoch_PSNR = epoch_PSNR / len(dataloader)
        epoce_mse_loss = epoce_mse_loss / len(dataloader)

        loss_history.append(epoch_loss)
        psnr_history.append(epoch_PSNR.cpu().detach().numpy())

        print(f"Epoch {epoch + 1}/{epochs} Loss: {epoch_loss:.6f} MSE: {epoce_mse_loss:.6f} PSNR: {epoch_PSNR:.4f} LR: {scheduler.get_last_lr()}")

        end_of_epoch(autoencoder, outputDirectory, epoch, loss_history, psnr_history, scheduler.get_last_lr())

        scheduler.step(epoch_loss)


 
def saveDecoderModelAsJSON(autoencoder, outputDirectory):
    #save model as json
    model_path = outputDirectory + 'decodermodel.json'
    model = autoencoder.decoder


    json_string = {}

    json_string['num_layers'] = len(model)
    for i, layer in enumerate(model):
        layer_category = str('layer' + str(i))
        layer_string = {}
        layer_string['name'] = layer.__class__.__name__

        if isinstance(layer, nn.Conv2d):
            layer_string['in_channels'] = layer.in_channels
            layer_string['out_channels'] = layer.out_channels
            layer_string['weight'] = layer.weight.data.flatten().cpu().numpy().tolist()
            layer_string['bias'] = layer.bias.data.flatten().cpu().numpy().tolist()

        json_string[layer_category] = layer_string
    

    with open(model_path, 'w') as f:
        json_string = json.dumps(json_string, indent=2)
        f.write(json_string)
