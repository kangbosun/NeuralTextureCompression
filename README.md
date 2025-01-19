# Neural Texture Compression

Neural Texture Compression Project
==========================

This study project implements a neural network-based approach for compressing and decompressing textures. The main components of the project include preprocessing, training an autoencoder, compressing textures, and decompressing textures.

https://research.nvidia.com/labs/rtr/neural_texture_compression/

Project Structure
-----------------
- AutoEncoder.py: Contains the implementation of the autoencoder model and related classes.
- Compressor.py: Contains functions for compressing and decompressing textures using the trained autoencoder.
- Run.py: Main script for running preprocessing, training, compression, and decompression tasks.
- requirements.txt: List of required Python packages.
- .gitignore: Specifies files and directories to be ignored by Git.

Setup Instructions
------------------
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required Python packages using the following command:
pip install -r requirements.txt


Features
------------------
AutoEncoder
Feature grids pyramid



Usage
-----
1. Preprocessing:
To preprocess the dataset, set `runPreprocessing = True` in `Run.py` or use the `-pp` command line flag.

2. Training:
To train the autoencoder, set `runAutoencoderTraining = True` in `Run.py` or use the `-train` command line flag. You can also specify the number of epochs using the `-e=<number_of_epochs>` flag.

3. Compression:
To compress textures, set `runCompression = True` in `Run.py` or use the `-comp` command line flag.

4. Decompression:
To decompress textures, set `runDecompression = True` in `Run.py` or use the `-decomp=<output_size>` command line flag.

5. Validation:
To validate the decompressed textures, set `runValidation = True` in `Run.py` or use the `-val` command line flag.

6. Cleaning Output:
To clean the output directory, set `cleanoutput = True` in `Run.py` or use the `-clean` command line flag.

ex.) py run.py -o=test1 -pp -train -e=10 -comp -decomp

Command Line Flags
------------------
- `-pp`: Run preprocessing.
- `-train`: Run autoencoder training.
- `-e=<number_of_epochs>`: Set the number of epochs for training.
- `-comp`: Run texture compression.
- `-decomp=<output_size>`: Run texture decompression with the specified output size.
- `-val`: Run validation.
- `-clean`: Clean the output directory.
- `-o=<workspace_directory>`: Set the workspace directory.


Contact
-------
For any questions or issues, please contact the project maintainer.
