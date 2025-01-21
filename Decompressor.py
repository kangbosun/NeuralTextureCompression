import torch
import numpy as np
from tqdm import tqdm

#decompress texture to channelarray
def decompressTextures(autoencoder, device, compressed_textures, outputSize, encoderSettings, batch_size=32):
    # Prepare autoencoder
    autoencoder.to(device)
    autoencoder.eval()

    # Log decompression general information
    print("Decompressing textures begin")
    print("Compressed Res:", len(compressed_textures[0]), "Channels:", len(compressed_textures[0][0][0]))
    print("Output Res    :", outputSize, "Channels:", encoderSettings.input_channels)

    # Create grid of normalized coordinates
    y_coords, x_coords = np.meshgrid(np.linspace(0, 1, outputSize), np.linspace(0, 1, outputSize), indexing='ij')

    y_coords = torch.Tensor(y_coords).to(device)
    x_coords = torch.Tensor(x_coords).to(device)

    # Gather compressed texture data and concatenate with normalized coordinates
    channels_tensor = None

    for i in range(0, len(compressed_textures)):
        #upscale feature grid
        grid = compressed_textures[i]
        grid = torch.tensor(grid).to(device).permute(2, 0, 1)
        grid = torch.nn.functional.interpolate(grid.unsqueeze(0), size=(outputSize, outputSize), mode='bilinear', align_corners=False)
        if channels_tensor is None:
            channels_tensor = grid
        else:
            channels_tensor = torch.cat([channels_tensor, grid], dim=1)

    inputs_tensor = torch.cat([channels_tensor, y_coords.unsqueeze(0).unsqueeze(0), x_coords.unsqueeze(0).unsqueeze(0)], dim=1)    
    
    with torch.no_grad():
        decoder_outputs = autoencoder.decoder(inputs_tensor)

    outputTexture = np.zeros((outputSize, outputSize, encoderSettings.input_channels), dtype=np.float32)    
    outputTexture[:] = decoder_outputs.squeeze(0).permute(1, 2, 0).cpu().numpy()

    print("Decompressing textures end")
    return outputTexture

