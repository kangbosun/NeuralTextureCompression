

# prerocess textures, combine all color channels into one array
import numpy as np
import tqdm

def preprocessTextures(textures, colorChannelCount):
    # Validate width/height from the first texture
    height = textures[0].height
    width = textures[0].width
    for texture in textures:
        if texture.width != width or texture.height != height:
            raise ValueError("All textures must have the same width and height.")
            
    channel_count_list = []
    for texture in textures:
        channel_count_list.append(texture.channels)

    # Concatenate along the channels dimension
    # Example: if you have 3 textures, each with shape (H, W, c1), (H, W, c2), (H, W, c3),
    # this will produce a shape (H, W, c1 + c2 + c3).
    colorChannels = np.concatenate(
        [tex.texture for tex in textures], 
        axis=-1
    ).astype(np.float32)
    
    # Optionally, check if the total channels match the expected count
    if colorChannels.shape[-1] != colorChannelCount:
        raise ValueError(
            f"Combined channels ({colorChannels.shape[-1]}) do not match "
            f"colorChannelCount ({colorChannelCount})."
        )
    
    return colorChannels, channel_count_list