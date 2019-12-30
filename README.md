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

