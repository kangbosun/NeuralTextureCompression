# load bit map image and extract rgb values
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image

#pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2

import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from multiprocessing import Pool
from multiprocessing import freeze_support

import Compressor
import Decompressor
import Preprocess
import TextureSet
import AutoEncoder



freeze_support()

#launch options
runPreprocessing = False
runAutoencoderTraining = False
runCompression = False
runValidation = False
runDecompression = False
cleanoutput = False


#set all true when debugging
#runPreprocessing = True
#runAutoencoderTraining = True
#runCompression = True
#runValidation = True

#input directory
inputDirectory = 'inputs\\'

#output directory
outputDirectory = 'outputs\\'

epochs = 10
decompress_size = 1024

num_feature_grids = 3

#init launch options from command line flags
import sys
for arg in sys.argv:
    if arg == '-pp':
        runPreprocessing = True
    if arg == '-train':
        runAutoencoderTraining = True
    if arg.startswith('-e='):
        epochs = int(arg[3:])        
    if arg == '-comp':
        runCompression = True
    if arg.startswith('-decomp='):
        decompress_size = int(arg[8:])
        runDecompression = True
    if arg.startswith('-featuregrids='):
        #clamp to 0-4
        num_feature_grids = max(0, min(4, int(arg[14:])))

    if arg == '-val':
        runValidation = True
    if arg == '-clean':
        cleanoutput = True

    #workspace directory
    if arg.startswith('-o='):
        outputDirectory = outputDirectory + arg[3:] + '\\' 
        print("Workspace directory: ", outputDirectory)

if cleanoutput:
    import shutil
    if os.path.exists(outputDirectory):
        shutil.rmtree(outputDirectory)


#make output directory
if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)

preprocessed_dataset_name = outputDirectory + "dataset" + '.npy'
channel_count_list_name = outputDirectory + "channel_count_list.npy"

dataSet = None
channel_count_list = []

print("-----------------------------")
if os.path.exists(preprocessed_dataset_name):
    
    print("Loading preprocessed data")
    dataSet = np.load(preprocessed_dataset_name)
    channel_count_list = np.load(channel_count_list_name)
    print("Preprocessed data found")
    
else:
    if runPreprocessing:
        # Load textures

        #get textures from input directory
        texturePaths = [inputDirectory + f for f in os.listdir(inputDirectory) if os.path.isfile(os.path.join(inputDirectory, f))]


        print("Loading textures")
        textures = TextureSet.loadTextures(texturePaths)
        print("Textures loaded: ", len(textures))

        TextureSet.PrintTextureInfo(textures)

        if (TextureSet.validateTextures(textures) == False):
            print("Textures are not valid")
            exit()

        # Get sum of color channel count
        colorChannelCount = TextureSet.GetCountOfColorChannels(textures)
        print("ColorChannelCount: ", colorChannelCount)

        #preprocess textures
        print("Preprocessing textures ", len(textures), " textures")
        dataSet, channel_count_list = Preprocess.preprocessTextures(textures, colorChannelCount)
        
        np.save(preprocessed_dataset_name, dataSet)
        print("Preprocessed textures saved to: ", preprocessed_dataset_name)

        #save channel count list
        np.save(channel_count_list_name, channel_count_list)
        print("Channel count list saved to: ", channel_count_list_name)

    else :
        print("Preprocessed textures not found, run preprocessing first with -pp flag")
        exit()
print("-----------------------------")

# Neural Compression

color_channel_count = dataSet.shape[2]
encoderSettings = AutoEncoder.EncoderSettings(color_channel_count, num_feature_grids)

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


#try load model
autoencoder = AutoEncoder.AutoEncoder(encoderSettings)

model = None
modelFilePath = outputDirectory + 'model.pth'

if os.path.exists(modelFilePath):
    model = torch.load(modelFilePath)
    print("Autoencoder found")

if model is not None:
    autoencoder.load_state_dict(model)
    print("Autoencoder loaded")


if runAutoencoderTraining:
    print("Training autoencoder")
    AutoEncoder.trainAutoEncoder(autoencoder, device, dataSet, epochs, 1024, 1e-4, outputDirectory)
    #save autoencoder
    torch.save(autoencoder.state_dict(), modelFilePath)

#compress textures
compressed_texture_size = 1024
compressed_texture_names = [outputDirectory + 'compressed0.tiff']
for i in range(1, num_feature_grids):
    compressed_texture_names.append(outputDirectory + 'compressed' + str(i) + '.tiff')


if runCompression:
    output_textures = Compressor.compressTextures(autoencoder, device, encoderSettings, dataSet, compressed_texture_size)

    #save compressed texture
    for i in range(len(output_textures)):
        output_texture  = output_textures[i]
        #save as tiff
        cv2.imwrite(compressed_texture_names[i], output_texture)
        print("Compressed texture saved to: ", compressed_texture_names[i])



    
#restore original textures from color channels
def restoreTextures(colorChannels, channel_count_list):
   # texure list
    textures = []

    #split color channels to textures
    channel_start = 0
    for i in range(len(channel_count_list)):
        channel_count = channel_count_list[i]
        texture = colorChannels[:, :, channel_start:channel_start+channel_count]
        textures.append(texture)

        channel_start += channel_count
        
    return textures

def saveRestoredTexture(texture, width, height, channelCount, name):
    data = None
    if channelCount == 1:
        data = np.array(texture).reshape(width, height)
    else:
        data = np.array(texture).reshape(width, height, channelCount)
    data = np.clip(data, 0, 1)
    data = (data * 255).astype(np.uint8)

    #convert BGR to RGB
    if channelCount == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    cv2.imwrite(name, data)

    if channelCount == 1:
        data = Image.fromarray(data, 'L')
    else:
        data = Image.fromarray(data, 'RGB')

    data.save(name)

decompressed_data_file_path = outputDirectory + 'decompressed.npy'
decompressed_data = None

if runDecompression:
    compressed_textures = []
    for compressed_texture_name in compressed_texture_names:
        compressed_texture = np.array(cv2.imread(compressed_texture_name, cv2.IMREAD_UNCHANGED), dtype=np.float16)
        compressed_textures.append(compressed_texture)

    decompressed_data = Decompressor.decompressTextures(autoencoder, device, compressed_textures, decompress_size, encoderSettings)
    np.save(decompressed_data_file_path, decompressed_data)

    #restore textures
    restoredTextures = restoreTextures(decompressed_data, channel_count_list)
    #save restored texture as bmp
    restoredTextureName = outputDirectory + 'restored'

    #save restored textures
    for i in range(len(channel_count_list)):
        saveRestoredTexture(restoredTextures[i], decompress_size, decompress_size, channel_count_list[i], restoredTextureName + str(i) + '.bmp')

if runValidation:
    if os.path.exists(decompressed_data_file_path):
        decompressedChannels = np.load(decompressed_data_file_path)
        

    #compare decompressed data with original data
    # print("Comparing decompressed data with original data")
    # print("Original data: ", dataSet.shape)
    # print("Decompressed data: ", decompressedChannels.shape)
    # #compare data with random 100 samples
    # print("Data comparison")
    # for i in range(100):
    #     x = np.random.randint(0, decompressSize)
    #     y = np.random.randint(0, decompressSize)
    #     print("Original: ", dataSet[y, x][0:3], "Decompressed: ", decompressedChannels[y, x][0:3])











