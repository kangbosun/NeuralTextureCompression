import os
from PIL import Image
import numpy as np
import tqdm
import cv2

# Texture paths array
# texturePaths = [
#     'inputs/PavingStones131_4K-Color.png', 
#     'inputs/PavingStones131_4K-NormalDX.png', 
#     'inputs/PavingStones131_4K-Roughness.png', 
#     'inputs/PavingStones131_4K-AmbientOcclusion.png']

# Class for each Texture
class Texture:
    def __init__(self, texturePath):
        self.texturePath = texturePath
        texture = cv2.imread(texturePath, cv2.IMREAD_UNCHANGED)
        self.texture = np.array(texture)
        self.texture = (texture / np.iinfo(self.texture.dtype).max)
        self.texture = np.clip(self.texture, 0, 1)

        print("shape: ", self.texture.shape)
    
        #check nan
        if np.isnan(self.texture).any():
            print("Texture contains nan values")

        # reshape the texture to height, width, channels
        self.texture = self.texture.reshape(self.texture.shape[0], self.texture.shape[1], -1)

        self.width = self.texture.shape[1]
        self.height = self.texture.shape[0]
        self.format = self.texture.dtype
        self.size = self.texture.size / (1024 * 1024)
        self.channels = self.texture.shape[2] if len(self.texture.shape) > 2 else 1

    def printTextureInfo(self):
        print("--------------------")
        print("Texture: ", self.texturePath)
        print("width: ", self.width, "height: ", self.height)
        print("format: ", self.format)
        print("size: ", self.size, "MB")
        print("channels: ", self.channels)
        print("--------------------")

def PrintTextureInfo(textureset):
    for texture in textureset:
        texture.printTextureInfo()


# Load textures with tqdm progress bar
def loadTextures(texturePaths):
    textures = []
    for texturePath in tqdm.tqdm(texturePaths):
        textures.append(Texture(texturePath))
    return textures

def GetCountOfColorChannels(textures):
    colorChannelCount = 0
    for texture in textures:
        colorChannelCount += texture.channels
    return colorChannelCount

def validateTextures(textures):
    # all textures must have the same width and height
    width = 0
    height = 0
    for texture in textures:
        if width == 0:
            width = texture.width
            height = texture.height
        elif width != texture.width or height != texture.height:
            # print error
            print("Textures must have the same width and height")
            print("Texture: ", texture.texturepath, " width: ", texture.width, " height: ", texture.height)
            return False