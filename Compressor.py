
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tqdm



from AutoEncoder import positional_encoding_xy
from AutoEncoder import EncoderSettings
from AutoEncoder import AutoEncoder


def compressTextures(autoencoder, device, encoderSettings, original_data, output_size, batch_size=32):
    import torch
    import numpy as np
    from tqdm import tqdm

    # Prepare autoencoder
    autoencoder.to(device)
    autoencoder.eval()

    #validatation of input data with encoder settings
    if len(original_data[0][0]) != encoderSettings.input_channels:
        print("Input data channels do not match encoder settings")
        return None
    
    input_channels_num = encoderSettings.input_channels
    output_channels_num = encoderSettings.channels_per_featuregrid
    output_textures_num = encoderSettings.featuregrids_num

    input_res = len(original_data[0])

    # Log compression general information
    print("Compressing textures begin")
    print("Input Res :", input_res, "Channels:", len(original_data[0][0]))
    print("Output Res:", output_size, "Channels:", output_channels_num)
    print("Output Textures:", output_textures_num)

    # cpu textures has H, W, C format
    outputTextures = []
    for i in range(output_textures_num):
        tex_size = max(256, output_size // (2 ** i))
        outputTextures.append(np.zeros((tex_size, tex_size, output_channels_num), dtype=np.float32))

    # Create grid of normalized coordinates
    y_coords, x_coords = np.meshgrid(np.linspace(0, 1, output_size), np.linspace(0, 1, output_size), indexing='ij')

    # Map to input texture coordinates
    ypos = (y_coords * (len(original_data) - 1)).astype(int)
    xpos = (x_coords * (len(original_data[0]) - 1)).astype(int)

    # Gather input texture data and concatenate with normalized coordinates
    selectedChannels = original_data[ypos, xpos]
    inputs = np.concatenate([selectedChannels, y_coords[..., None], x_coords[..., None]], axis=-1)

    # Convert inputs to tensor
    inputs_tensor = torch.Tensor(inputs).to(device)
    inputs_tensor = inputs_tensor.permute(2, 0, 1)

    with torch.no_grad():
        encoder_outputs = autoencoder.encoder(inputs_tensor)

    grid_size = output_size
    for i in range(0, output_textures_num):
        #scale down output data
        grid_data = encoder_outputs[i*output_channels_num: (i+1)*output_channels_num].unsqueeze(0)
        scaled_texture = torch.nn.functional.interpolate(grid_data, size=(grid_size, grid_size), mode='bilinear')
        outputTextures[i][:] = scaled_texture.squeeze(0).permute(1, 2, 0).cpu().numpy()
        grid_size = max(256, grid_size // 2)

    print("Compressing textures end")
    return outputTextures


#Compress featuregrids to BC6H with mipmaps
texConv_path = "bin\\texconv.exe"

def compressToBC6H(input_texture, output_texture):
    import os
    import subprocess

    #create mipmaps
    mipmaps = "-m " + str(0)
    command = texConv_path + " -nologo -y -f BC6H_SF16 -o " + output_texture + " " + mipmaps + " " + input_texture
    print(command)
    os.system(command)