# VGG-19-FeedForward_CUDA
 
Prerequisite:
Download the Weights and Biases for the layer from the link posted below:
https://drive.google.com/open?id=1bq95bj-ToTRPsihwxwDvefUWBcK-o1L9

The test images are also provided with the drive link.
Save the images and weights in the same folder as the code.

Implementation:
make all

or

Compile using nvcc -c kernel.cu 
nvcc -o kernel kernel.o
