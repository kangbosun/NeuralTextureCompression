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

    # Map to input compressed texture coordinates
    ypos = (y_coords * (len(compressed_textures[0]) - 1)).astype(int)
    xpos = (x_coords * (len(compressed_textures[0][0]) - 1)).astype(int)

    # Gather compressed texture data and concatenate with normalized coordinates
    selectedChannels = compressed_textures[0][ypos, xpos]
    for i in range(1, len(compressed_textures)):
        scaled_ypos = (y_coords * (len(compressed_textures[i]) - 1)).astype(int)
        scaled_xpos = (x_coords * (len(compressed_textures[i][0]) - 1)).astype(int)
        selectedChannels = np.concatenate([selectedChannels, compressed_textures[i][scaled_ypos, scaled_xpos]], axis=-1)
    
    inputs = np.concatenate([selectedChannels, y_coords[..., None], x_coords[..., None]], axis=-1)

    # Convert inputs to tensor
    inputs_tensor = torch.Tensor(inputs.astype(np.float32)).to(device)

    inputs_tensor = inputs_tensor.permute(2, 0, 1)

    with torch.no_grad():
        decoder_outputs = autoencoder.decoder(inputs_tensor)

    outputTexture = np.zeros((outputSize, outputSize, encoderSettings.input_channels), dtype=np.float32)    
    outputTexture[:] = decoder_outputs.permute(1, 2, 0).cpu().numpy()

    print("Decompressing textures end")
    return outputTexture

