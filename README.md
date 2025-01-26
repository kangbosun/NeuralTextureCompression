# Neural Texture Compression

Neural Texture Compression Project
==========================

This study project implements a neural network-based approach for compressing and decompressing textures. The main components of the project include preprocessing, training an autoencoder, compressing textures, and decompressing textures.

https://research.nvidia.com/labs/rtr/neural_texture_compression/

https://www.gdcvault.com/play/1034892/Machine-Learning-Summit-Real-time

Used assets are from https://ambientcg.com/view?id=PavingStones131

Setup Instructions
------------------
1. Install pytorch (requires CUDA version)  https://pytorch.org/get-started/locally/
2. Clone the repository to your local machine.
3. Navigate to the project directory.
4. Install the required Python packages using the following command:
`pip install -r requirements.txt`

Sample Outputs
--------------
Using the trained neural network, four 4K textures (with 3, 3, 1, and 1 channels) are compressed into four feature grids (top resolution 1024, saved as .jxr) and then decompressed back into their original form (restored[n].bmp).

The training process takes approximately 15 minutes on an RTX 4060.

For real-time rendering, .dds files are used, leveraging BC compression with hardware acceleration.

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

5. Feature Grids:
The -featuregrids=[1-4] flag specifies the number of feature grid levels used for compression and decompression. Higher values result in better reconstruction quality but require more storage. For example, -featuregrids=4 uses a four-level feature grid pyramid.

6. Cleaning Output:
To clean the output directory, set `cleanoutput = True` in `Run.py` or use the `-clean` command line flag.

ex.) `py run.py -o=test1 -pp -train -e=10 -comp -decomp=4096 -featuregrids=4`



Command Line Flags
------------------
- `-pp`: Run preprocessing.
- `-train`: Run autoencoder training.
- `-e=<number_of_epochs>`: Set the number of epochs for training.
- `-comp`: Run texture compression.
- `-decomp=<output_size>`: Run texture decompression with the specified output size.
- `-featuregrids=[1-4]`: Set the number of feature grid levels used for compression and decompression.
- `-clean`: Clean the output directory.
- `-o=<workspace_directory>`: Set the workspace directory.

Related
------------------
AutoEncoder

Feature grids pyramid

Contact
-------
For any questions or issues, please contact the project maintainer.
