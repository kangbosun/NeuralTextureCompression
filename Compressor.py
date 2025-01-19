
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tqdm


from AutoEncoder import positional_encoding_xy
from AutoEncoder import EncoderSettings
from AutoEncoder import AutoEncoder



#compress textures
def compressTextures(autoencoder, device, encoderSettings, colorChannels, outputSize, batch_size=32):
    # Prepare autoencoder
    autoencoder.to(device)
    autoencoder.eval()

    # Log compression general information
    print("Compressing textures begin")
    print("Input Res :", len(colorChannels[0]), "Channels:", len(colorChannels[0][0]))
    print("Output Res:", outputSize, "Channels:", encoderSettings.outputChannels)

    outputTexture = np.zeros((outputSize, outputSize, encoderSettings.outputChannels), dtype=np.float32)

    progress_bar = tqdm.tqdm(total=outputSize, desc="Compressing textures")
    # Create grid of normalized coordinates
    for y in range(outputSize):
        for x in range(outputSize):
            y_coords = y / (outputSize - 1)
            x_coords = x / (outputSize - 1)

            ypos = int(y_coords * (len(colorChannels) - 1))
            xpos = int(x_coords * (len(colorChannels[0]) - 1))

            selectedChannels = colorChannels[ypos, xpos]
            combined = np.concatenate([selectedChannels, [y_coords, x_coords]], axis=-1)
            encoder_inputs = torch.Tensor(combined.astype(np.float32)).to(device)

            with torch.no_grad():
                encoder_outputs = autoencoder.encoder(encoder_inputs)
                outputTexture[y, x] = encoder_outputs[:-2].cpu().numpy()
                
                #exit app if we found a negative or over 1 value
                if np.any(outputTexture[y, x] < 0):
                    print("Negative value found")
                    print(outputTexture[y, x])
                    exit()

        progress_bar.update(1)

    print("Compressing textures end")
    return outputTexture

def compressTextures2(autoencoder, device, encoderSettings, original_data, output_size, batch_size=32):
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

    # Log compression general information
    print("Compressing textures begin")
    print("Input Res :", len(original_data[0]), "Channels:", len(original_data[0][0]))
    print("Output Res:", output_size, "Channels:", output_channels_num)
    print("Output Textures:", output_textures_num)

    # cpu textures has H, W, C format
    outputTextures = []
    for i in range(output_textures_num):
        tex_size = output_size // (2 ** i)
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


    data_loader = torch.utils.data.DataLoader(inputs_tensor, batch_size=2, shuffle=False)
    
    current_idx = 0
    #torch_output = torch.zeros(output_channels_num * output_textures_num, output_size, output_size, device=device)
    with torch.no_grad():
        encoder_outputs = autoencoder.encoder(inputs_tensor)
        encoder_outputs

        print(encoder_outputs.shape)

    #fine featuregrid
    outputTextures[0][:] = encoder_outputs[:output_channels_num].permute(1, 2, 0).cpu().numpy()

    scale_factor = 0.5
    for i in range(1, output_textures_num):
        #scale down output data
        coarse_data = encoder_outputs[i*output_channels_num: (i+1)*output_channels_num].unsqueeze(0)
        scaled_texture = torch.nn.functional.interpolate(coarse_data, scale_factor=scale_factor, mode='bilinear')
        outputTextures[i][:] = scaled_texture.squeeze(0).permute(1, 2, 0).cpu().numpy()
        scale_factor *= 0.5

    print("Compressing textures end")
    return outputTextures

    # Process in batches
    with torch.no_grad():
        outputs_tensor = torch.zeros((max_height, max_width, output_channels_num * output_textures_num), device=device)
        for start_idx in tqdm(range(0, outputSize * outputSize, batch_size), desc="Compressing textures"):
            end_idx = min(start_idx + batch_size, outputSize * outputSize)

            # Flatten the batch for processing
            batch = inputs_tensor.view(-1, inputs_tensor.shape[-1])[start_idx:end_idx]

            # Forward pass through the autoencoder
            outputs_batch = autoencoder.encoder(batch)

            # Store the results
            outputs_tensor.view(-1, encoderSettings.outputChannels)[start_idx:end_idx] = outputs_batch

        # Reshape the results into the output texture
        outputTexture = np.zeros((outputSize, outputSize, 3), dtype=np.float32)
        outputTexture[:] = outputs_tensor.view(outputSize, outputSize, -1).cpu().numpy()

    print("Compressing textures end")
    return outputTexture

#decompress texture to channelarray
def decompressTextures(autoencoder, device, compressed_textures, outputSize, encoderSettings, batch_size=32):
    import torch
    import numpy as np
    from tqdm import tqdm

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

    # Process in batches
    # with torch.no_grad():
    #     outputs_tensor = torch.zeros((outputSize, outputSize, encoderSettings.input_channels), device=device)
    #     for start_idx in tqdm(range(0, outputSize * outputSize, batch_size), desc="Decompressing textures"):
    #         end_idx = min(start_idx + batch_size, outputSize * outputSize)

    #         # Flatten the batch for processing
    #         batch = inputs_tensor.view(-1, inputs_tensor.shape[-1])[start_idx:end_idx]

    #         # Forward pass through the autoencoder decoder
    #         outputs_batch = autoencoder.decoder(batch)

    #         # Store the results
    #         outputs_tensor.view(-1, encoderSettings.input_channels)[start_idx:end_idx] = outputs_batch

    #     # Reshape the results into the decompressed texture
    #     outputTexture = np.zeros((outputSize, outputSize, encoderSettings.input_channels), dtype=np.float32)
    #     outputTexture[:] = outputs_tensor.view(outputSize, outputSize, -1).cpu().numpy()

    print("Decompressing textures end")
    return outputTexture



