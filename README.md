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


## Motivation
Convolutional Neural Networks are a category of neural network which are used in areas such as image
classification, image segmentation, instance segmentation, activity recognition etc. ConvNets have
been successful in identifying faces, objects and traffic signs apart from powering vision in robots and
self-driving cars. The biggest drawback of using CNNs is that the training requires very large datasets
to converge to their global optima and the huge networks take long time, even for inferencing.
Presently, we can see several companies working on building AI chips using CNNs, RNNs, LSTM etc.
These include big names like Nvidia, Intel, Google, Qualcomm, AMD, Microsoft and several start-ups
like Cerebras, Habana Labs, Sambanova Systems, Graphcore etc. The AI chips being developed are
used for either training a new network or for inference on an already trained network, using the
weights and biases values from training performed in the past. The tasks that these AI chips are going
to be used for very critical for example self-driving cars, where we need super quick response from
the networks to decide the course of action. Thus, there is a need to speed up the inference to make
these chips run faster.
A Convolutional Neural Network majorly consists of convolutional layers and fully connected layers,
with max pool, average pool, batch normalization and activation functions. Considering the AlexNet
as an example, almost 95% of the GPU time and 89% of the CPU time is spent on proceeding through
the convolutional layers and fully connected layers. This creates the need to find ways to optimize the
operations going on in these layers. To do so, we can leverage the decades of research done to
optimize matrix-matrix multiplication. The cuBLAS library has CUDA GEMM API, Intel MKL has special
optimized CPU GEMM and for devices supporting OpenCL we have ciBLAS GEMM API. In this project,
we focus on learning the problems faced while developing these optimized libraries. Another
motivation comes from the fact that the hardware AI chips being built don’t have these libraries built
into them. Hence the programmers need to perform all operations themselves and in an optimized
manner.

## Approach
In this project, we focus on parallelizing and hence speeding up the feedforward of CNNs using a
parallel computing architecture using CUDA. We are going to convert convolutions into general matrix
multiplications and exploit parallelism at each layer in the forward propagation to improve memory
efficiency and achieve a lower execution time.
GEMM or General Matrix to Matrix Multiplication, as the name suggests is the task of multiplying two
matrices to give the final output. We use a function called “im2col” to first convert the convolution
operation into simple GEMM operation.
For any convolutional layer, we extract filter sized image patches, with given stride length and convert
them into columns. We also take care of the channels across the input to the layers, so the channels
come one below the other in a column. In order to maintain the operation of convolution as expected,
we convert the filters into rows, which get multiplied by the columns of im2col. The channels in the
filter are hence placed (channel-wise) one after the other in the row. The following figures show how
we convert the convolution operation into GEMM operation:

![image](https://user-images.githubusercontent.com/16237584/71584016-a269fb80-2ac5-11ea-8bde-d8a4089c416b.png)

![image](https://user-images.githubusercontent.com/16237584/71583928-4acb9000-2ac5-11ea-8143-8e60cb7d750c.png)

![image](https://user-images.githubusercontent.com/16237584/71583968-6c2c7c00-2ac5-11ea-985f-a52abed7cea5.png)

![image](https://user-images.githubusercontent.com/16237584/71583986-81a1a600-2ac5-11ea-9c55-9204202a986a.png)

Image Reference:
[1] https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

The fully connected layer directly can be implemented with a GEMM operation, so we do not need
any im2col in this case.
This method may seem inefficient in the terms of the expansion of memory size as we are duplicating
the overlapping pixels for consecutive convolutions with the kernel. However, the advantages in terms
of memory accesses highly overweighs the wastage incurred in terms of this data redundancy.
We will be implementing the complete CNN and convert convolutional and fully connected layers into
GEMM operation and parallelize them using CUDA.

## Model and Dataset chosen
For the project we have chosen ImageNet with 1000 classes dataset and the VGG-19 model. The
weights and biases for the trained model are available and we are using them to run the parallelized
feedforward with GEMM.

![image](https://user-images.githubusercontent.com/16237584/71584209-887ce880-2ac6-11ea-8ad6-edee54fcf5a5.png)

VGG stands for the “Visual Geometry Group”. It is a group at the University of Oxford and is a very
popular network for image classification and is further used as backbone for activity recognition,
image segmentation, instance segmentation and many other tasks. The main fundamental behind
VGG is that by decreasing the size of kernels and repeating them several times, we can observe same
effect as a larger kernel and decrease the number of parameters. This also allows to make the network
deeper. There are several variations of VGG network which add more layers of convolutions or
regularization like batch normalization or dropout. A major advantage of using VGG networks is that
while testing the fully connected layers can be replaced by convolution layers which allows any image
size allowed as input to the trained network.
Currently, we chose VGG-19 network for our inference operation. It contains convolutional layers with
ReLu activation, Maxpool layers and Fully Connected layers at the end. The VGG-19 network has 46
layers in total and a parameter count of 143,667,240.

## Implementation steps
1. Download all weights and biases for all layers from Kaggle website corresponding to the
ImageNet dataset for VGG-19.
2. Use Python to extract weights and biases and store these values in different text files. Also
make a text file with dimensions of weights and biases in different layers.
3. When running the inference on the network, we first read the RGB image into an allocated
memory. We then send this to the GPU memory.
4. We pad it with appropriate zeros. For padding, we parallelize padding operation such that
padding for one channel is done by one thread. So, the number of threads used for padding
will be equal to the number of channels as input to the layer.
5. Once the padding is done, we free the memory allocated to the input image from CPU and
GPU.
6. Then we pass the padded image within the GPU to im2col function, which converts it to a
matrix as mentioned earlier. Here, the parallelization is as follows: “Rows corresponding to
one complete horizontal movement of the kernel from left to right were assigned to one
thread.” So, total number of threads used will be equal to the output size after convolution.
Each thread is responsible for all values across channels as well.
7. Then we free the memory GPU memory for padded image.
8. We now read the dimensions of the weights and biases for the layer from the dimensions text
file that we had stored earlier. Then allocate memory of dimension size for weights and biases
in the CPU. We then transfer these weights and biases into the GPU.
9. Now, GEMM convolution is performed with the im2col of the image and the weights and
biases, all present in the GPU. Here, parallelization is done in a way that each channel of
output is computed by a thread. So, the number of threads will be equal to the number of
output channels. ReLU, which is performed after each convolution layer is integrated with the
GEMM operation itself, where max(0,x) is applied when storing the output.
10. Once the GEMM operation is completed and we have the convolution output, we free the
GPU memory for im2col image in GPU and weights and biases in both CPU and GPU.
11. The output from this layer can be fed to:
a. A convolution layer which pads the input, performs im2col, reads weights and biases
into CPU, transfers to GPU and finally performs GEMM with ReLU.
b. A maxpool layer, which calculates the maximum of the kernel that goes around the
image. Parallellization in this case is that each channel is covered by one thread. So,
the number of threads involved is equal to the number of input channels.
12. Again, once the output of previous layer is consumed, we free the CPU and GPU memory
instantly to minimize memory usage.
13. When a fully connected layer is encountered, we again read the weights and biases
dimensions and allocate CPU memory to read them from text file. We read these values and
then transfer to GPU memory. For a FC layer, we directly perform matrix multiplication on the
input with the weight values we read and add biases accordingly. For a FC layer parallelization
is such that each thread handles one output node. If the output nodes are more than 1024,
we assign (total number of nodes / 1024) nodes to each thread. Hence, in the FC0 and FC1
layers, where we have 4096 nodes in output, each thread handles 4 nodes but in last layer
where we have 1000 outputs, we only assign one node to each thread.
14. The Softmax layer is removed as we are performing inference and hence, we can just take the
maximum of the final layer output and can avoid the unnecessary computation for exponent
of output and dividing by sum of all such values and repeating the same for all output values.
