#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <algorithm>
#include <cmath>
#include<opencv2/opencv.hpp>
#include <time.h> 
#define input_file "purse.jpg"
using namespace std;
using namespace cv;
#define h 224
#define w 224
//#define kernel_size 3
string dimensions = "dimensions.txt";
string layer0 = "layer0.txt";
string layer1 = "layer1.txt";
string layer2 = "layer2.txt";
string layer3 = "layer3.txt";
string layer4 = "layer4.txt";
string layer5 = "layer5.txt";
string layer6 = "layer6.txt";
string layer7 = "layer7.txt";
string layer8 = "layer8.txt";
string layer9 = "layer9.txt";
string layer10 = "layer10.txt";
string layer11 = "layer11.txt";
string layer12 = "layer12.txt";
string layer13 = "layer13.txt";
string layer14 = "layer14.txt";
string layer15 = "layer15.txt";
string fclayer0 = "fclayer0.txt";
string fclayer1 = "fclayer1.txt";
string fclayer2 = "fclayer2.txt";


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///Kernel function to convert from image to column
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void im2col(int* r, int* g, int* b, double* result, int cols)
{

	int k;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	k = int(idx * 27 * (cols - 2));
	//printf("%d\n", result[k]);
	//printf("\nhello id %d Kvalue: %d", idx, k);

	//printf("GPU %d", b[1000]);
	for (int j = 1; j < cols - 1; j++)
	{

		for (int m = -1; m <= 1; m++)
		{
			//printf("\nhello id %d Kvalue: %d j value: %d", idx, k,j);
			result[k] = double(r[cols * (idx + 1 + m) + j - 1]);
			result[k + 1] = double(r[cols * (idx + 1 + m) + j]);
			result[k + 2] = double(r[cols * (idx + 1 + m) + j + 1]);
			k = k + 3;
		}
		//printf("idx: %d j: %d\n",idx,j);
		for (int m = -1; m <= 1; m++)
		{
			result[k] = double(g[cols * (idx + 1 + m) + j - 1]);
			result[k + 1] = double(g[cols * (idx + 1 + m) + j]);
			result[k + 2] = double(g[cols * (idx + 1 + m) + j + 1]);
			k = k + 3;
		}
		for (int m = -1; m <= 1; m++)
		{
			result[k] = double(b[cols * (idx + 1 + m) + j - 1]);
			result[k + 1] = double(b[cols * (idx + 1 + m) + j]);
			result[k + 2] = double( b[cols * (idx + 1 + m) + j + 1]);
			k = k + 3;
		}

	}
	
	//printf("hello id %d", idx);
}


__global__ void input2col(double* input, int input_size, double* result, int input_channels, int output_size, int stride) //input, insize,output,no_in_ch,no)out_size,stride
{
	int k;
	int kernel_size = 3;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	k = int(idx * input_channels * (kernel_size* kernel_size) * (output_size));
	//printf("%d\n", result[k]);
	for (int j = 1; j < input_size - 1; j = j + stride)
	{
		for (int i = 0; i < input_channels; i++)
		{
			for (int m = -1; m <= 1; m++)
			{
				result[k] = input[i * (input_size * input_size) + input_size * (idx + 1 + m) + j - 1];
				result[k + 1] = input[i * (input_size * input_size) + input_size * (idx + 1 + m) + j];
				result[k + 2] = input[i * (input_size * input_size) + input_size * (idx + 1 + m) + j + 1];
				k = k + 3;
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///CUDA Function for padding
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void input_pad(double* input, int input_size, double* output, int pad_size) // input,in_size, output, padsize
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;// for each row
	int output_size = input_size + 2 * pad_size;
	for (int i = pad_size; i < input_size + pad_size; i++)
	{
		for (int j = pad_size; j < input_size + pad_size; j++)
		{
			output[idx * (output_size * output_size) + i * output_size + j] = input[idx * (input_size * input_size) + (i - pad_size) * input_size + (j - pad_size)];
		}
	}
	//printf("\nThread id %d", idx);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///CUDA Function for Forward Propagation of layer
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void weight_mult(double* W, double* B, double* input, int input_channels, int output_size, double* l_output)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;// for each row
	int kernel_size = 3;
	for (int i = 0; i < output_size; i++)
	{
		for (int j = 0; j < output_size; j++)
		{
			for (int k = 0; k < (kernel_size *kernel_size ) * input_channels; k++)
			{
				l_output[idx * (output_size *output_size ) + output_size * i + j] += W[(kernel_size * kernel_size) * input_channels * idx + k] * input[i * (kernel_size * kernel_size) * input_channels * output_size + j * (kernel_size * kernel_size) * input_channels + k];
			}
			l_output[idx * (output_size * output_size) + output_size * i + j] += B[idx];

			if (l_output[idx * (output_size * output_size) + output_size * i + j] < 0)
				l_output[idx * (output_size * output_size) + output_size * i + j] = 0;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///CUDA Function for maxpool
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void maxpool(double* input, int input_size, double* m_output, int output_size, int stride)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int val[4];
	int minval = INT_MIN;
	//printf("%d\n", result[k]);
	for (int i = 0; i < input_size - 1; i = i + stride)
	{
		for (int j = 0; j < input_size - 1; j = j + stride)
		{
			val[0] = input[idx * (input_size * input_size) + i * input_size + j];
			val[1] = input[idx * (input_size * input_size) + i * input_size + j + 1];
			val[2] = input[idx * (input_size * input_size) + (i + 1) * input_size + j];
			val[3] = input[idx * (input_size * input_size) + (i + 1) * input_size + j + 1];
			for (int k = 0; k < 4; k++)
			{
				if (minval < val[k])
				{
					minval = val[k];
				}
			}
			m_output[int(idx * ((output_size)* output_size) + (i / 2) * (output_size)+(j / 2))] = double(minval);
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///CUDA Function for Fully Connected layer
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void fc_mult(double* input, double* W, double* B, int input_size, int output_size, double* output)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;// for each row

	for (int i = 0; i < floorf(output_size / 1024); i++)
	{
		output[idx + i * 1024] = 0;
		for (int j = 0; j < input_size; j++)
		{
			output[idx + i * 1024] += input[j] * W[idx + i * 1024 + j * output_size];
		}
		output[idx + i * 1024] += B[idx + i * 1024];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///Function to read Dimension
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int readdim(double* dim)
{
	FILE* fp;
	ifstream myReadFile;
	myReadFile.open(dimensions);
	if (!myReadFile)
	{
		cout << "Unable to open file";
		exit(1); // terminate with error
	}
	int k = 0;
	if (myReadFile.is_open())
	{
		while (!myReadFile.eof())
		{
			myReadFile >> dim[k];
			k++;
		}
	}
	myReadFile.close();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Function to read Weights and biases
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int read_model_params(string layername, double* W, double* B, int sizeofW)
{
	FILE* fp;
	ifstream myReadFile;
	myReadFile.open(layername);
	if (!myReadFile)
	{
		cout << "Unable to open file";
		exit(1); // terminate with error
	}
	int k;
	if (myReadFile.is_open())
	{
		k = 0;
		while (!myReadFile.eof() && k < sizeofW)
		{
			myReadFile >> W[k];
			k++;
		}
		k = 0;
		while (!myReadFile.eof())
		{
			myReadFile >> B[k];
			k++;
		}
	}
	myReadFile.close();
}

int main()
{
	time_t my_time = time(NULL);
	cout << "Starting Time: " << ctime(&my_time) << endl;
	///Read Dimensions FIle
	double* dim = (double*)calloc(4 * 16 + 2 * 3, sizeof(double));
	readdim(dim);


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Set the dimensions to the Weights and Biases
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int* sizeof_CNN_W = (int*)calloc(16, sizeof(int));
	int* sizeof_CNN_B = (int*)calloc(16, sizeof(int));
	int* sizeof_FC_W = (int*)calloc(3, sizeof(int));
	int* sizeof_FC_B = (int*)calloc(3, sizeof(int));
	int k = 0;
	for (int i = 0; i < 16; i++)
	{
		sizeof_CNN_W[i] = (int)(dim[k] * dim[k + 1] * dim[k + 2] * dim[k + 3]);
		sizeof_CNN_B[i] = (int)(dim[k + 3]);
		k = k + 4;
	}

	for (int i = 0; i < 3; i++)
	{
		sizeof_FC_W[i] = (int)(dim[k] * dim[k + 1]);
		sizeof_FC_B[i] = (int)(dim[k + 1]);
		k = k + 2;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// initialize the Weights and Biases of every Layer
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	double* W0, * W1, * W2, * W3, * W4, * W5, * W6, * W7, * W8, * W9, * W10, * W11, * W12, * W13, * W14, * W15;
	double* B0, * B1, * B2, * B3, * B4, * B5, * B6, * B7, * B8, * B9, * B10, * B11, * B12, * B13, * B14, * B15;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/* for (int i = 0; i < sizeof_CNN_W[0]; i++)
	{
	cout << W0[i]<<endl;
	}
	cout << "\n\n\n Biasesn\n\n\n" << endl;
	for (int i = 0; i < sizeof_CNN_B[0]; i++)
	{
	cout << B0[i] << endl;
	}
	*/
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Image to Column part
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Reading the image and convert to multiple channels R,G,B

	Mat img = imread(input_file);
	Mat imgpad;
	imgpad.create(img.rows + 2 * 1, img.cols + 2 * 1, img.type());
	imgpad.setTo(cv::Scalar::all(0));
	img.copyTo(imgpad(Rect(1, 1, img.cols, img.rows)));
	img.release();

	int* r = (int*)calloc(imgpad.rows * imgpad.cols, sizeof(int));
	int* g = (int*)calloc(imgpad.rows * imgpad.cols, sizeof(int));
	int* b = (int*)calloc(imgpad.rows * imgpad.cols, sizeof(int));

	unsigned char* input = (unsigned char*)(imgpad.data);
	k = 0;
	int rows = imgpad.rows;
	int cols = imgpad.cols;
	for (int j = 0; j < rows; j++)
	{
		for (int i = 0; i < cols; i++)
		{
			b[k] = input[imgpad.step * j + i];
			g[k] = input[imgpad.step * j + i + 1];
			r[k] = input[imgpad.step * j + i + 2];
			k++;
		}
	}
	imgpad.release();

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-0 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	my_time = time(NULL);
	cout << "Executing Layer 0: " << ctime(&my_time) << endl;
	///Set the parameters: no of channels, stride, kernel size, output image size
	int kernel_size = 3;
	int stride = 1;
	int input_channels = 3;
	int input_size = rows;
	int output_channels = 64;
	int output_size = floor((input_size - kernel_size) / stride) + 1;

	//Set the CUDA params, Allocate the GPU memory and Copy the data from CPU to GPU
	dim3 gridDim_l0(1);
	dim3 blockDim_l0(output_size);
	//cout << out_size * out_size * kernel_size * kernel_size * no_of_channels<<endl;
	int* gpu_r, * gpu_g, * gpu_b;
	double* gpu_im2col_l0;
	cudaError_t Status;
	
	cudaMalloc((void**)& gpu_r, input_size * input_size * sizeof(int));
	cudaMalloc((void**)& gpu_g, input_size * input_size * sizeof(int));
	cudaMalloc((void**)& gpu_b, input_size * input_size * sizeof(int));

	//cout << "\n im2col size "<<pow(output_size,2)* pow(kernel_size,2)*input_channels<<endl;
	cudaMalloc((void**)& gpu_im2col_l0, pow(output_size, 2)* pow(kernel_size, 2)* input_channels * sizeof(double));
	cudaMemcpy(gpu_r, r, input_size * input_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_g, g, input_size * input_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b, input_size * input_size * sizeof(int), cudaMemcpyHostToDevice);

	//cudaError errcode = cudaGetLastError();
	//cout << endl<< cudaGetErrorString(errcode);
	//Call the im2col kernel function

	//copy from GPU to CPU


	
	im2col << < gridDim_l0, blockDim_l0 >> > (gpu_r, gpu_g, gpu_b, gpu_im2col_l0, input_size);
	cudaDeviceSynchronize();

	double* result = (double*)calloc(pow(output_size, 2) * pow(kernel_size, 2) * input_channels, sizeof(double));
	cudaMemcpy(result, gpu_im2col_l0, pow(output_size, 2)* pow(kernel_size, 2)* input_channels * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();






	// free unncessary memory
	free(r);
	free(g);
	free(b);
	free(result);
	cudaFree(gpu_r);
	cudaFree(gpu_g);
	cudaFree(gpu_b);

	//Perform convolution for the layer

	//Read the weights and biases
	W0 = (double*)calloc(sizeof_CNN_W[0], sizeof(double));
	B0 = (double*)calloc(sizeof_CNN_B[0], sizeof(double));
	read_model_params(layer0, W0, B0, sizeof_CNN_W[0]);

	//Alocate memory for output
	double* gpu_W0, * gpu_B0, * gpu_output_l0;
	cudaMalloc((void**)& gpu_W0, sizeof_CNN_W[0] * sizeof(double));
	cudaMalloc((void**)& gpu_B0, sizeof_CNN_B[0] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l0, pow(output_size,2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W0, W0, sizeof_CNN_W[0] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B0, B0, sizeof_CNN_B[0] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W0(1);
	dim3 blockDim_W0(output_channels);
	weight_mult << < gridDim_W0, blockDim_W0 >> > (gpu_W0, gpu_B0, gpu_im2col_l0, input_channels, output_size, gpu_output_l0);
	cudaDeviceSynchronize();


	free(W0);
	free(B0);

	cudaFree(gpu_W0);
	cudaFree(gpu_B0);
	cudaFree(gpu_im2col_l0);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-1 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 1: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 64;
	int pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l1;
	cudaMalloc((void**)& gpu_input_l1, pow((input_size + 2 * pad_size) , 2) * input_channels * sizeof(double));
	dim3 gridDim_l1(1);
	dim3 blockDim_l1(input_channels);

	input_pad << < gridDim_l1, blockDim_l1 >> > (gpu_output_l0, input_size, gpu_input_l1, pad_size); // input,in_size, output, padsize
	//cudaError err = cudaGetLastError();
	//result = (double*)calloc(pow((input_size + 2 * pad_size), 2) * input_channels, sizeof(double));
	//cudaMemcpy(result, gpu_input_l1, pow((input_size + 2 * pad_size), 2)* input_channels * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//err = cudaGetLastError();
	//cout << endl << cudaGetErrorString(err) << endl;
	//for (int i = h * w / 4; i < h * w / 4 + 100; i++)
		//cout << result[i] << endl;
	
	cudaFree(gpu_output_l0);



	// convert to columns using im2col
	double* gpu_im2col_l1;
	cudaMalloc((void**)& gpu_im2col_l1, pow(output_size , 2) * pow(kernel_size, 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I1(1);
	dim3 blockDim_I1(output_size);
	input2col << < gridDim_I1, blockDim_I1 >> > (gpu_input_l1, input_size, gpu_im2col_l1, input_channels, output_size, stride);
	



	//Perform convolution for the layer

	//Read the weights and biases
	W1 = (double*)calloc(sizeof_CNN_W[1], sizeof(double));
	B1 = (double*)calloc(sizeof_CNN_B[1], sizeof(double));
	read_model_params(layer1, W1, B1, sizeof_CNN_W[1]);

	//Alocate memory for output
	double* gpu_W1, * gpu_B1, * gpu_output_l1;
	cudaMalloc((void**)& gpu_W1, sizeof_CNN_W[1] * sizeof(double));
	cudaMalloc((void**)& gpu_B1, sizeof_CNN_B[1] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l1, pow(output_size, 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W1, W1, sizeof_CNN_W[1] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B1, B1, sizeof_CNN_B[1] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W1(1);
	dim3 blockDim_W1(output_channels);
	weight_mult << < gridDim_W1, blockDim_W1 >> > (gpu_W1, gpu_B1, gpu_im2col_l1, input_channels, output_size, gpu_output_l1);
	free(W1);
	free(B1);
	cudaFree(gpu_input_l1);
	cudaFree(gpu_W1);
	cudaFree(gpu_B1);
	cudaFree(gpu_im2col_l1);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// MAXPOOL0 /////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//set the parameters
	kernel_size = 2;
	stride = 2;
	input_size = output_size;
	output_size = int(input_size / 2);
	input_channels = output_channels;
	output_channels = input_channels;
	double* gpu_output_m0;
	cudaMalloc((void**)& gpu_output_m0, pow(output_size,2) * output_channels * sizeof(double));

	dim3 gridDim_M0(1);
	dim3 blockDim_M0(output_channels);
	maxpool << < gridDim_M0, blockDim_M0 >> > (gpu_output_l1, input_size, gpu_output_m0, output_size, stride);
	cudaFree(gpu_output_l1);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-2 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 2: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 128;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l2;
	cudaMalloc((void**)& gpu_input_l2, pow((input_size + 2 * pad_size) , 2) * input_channels * sizeof(double));
	dim3 gridDim_l2(1);
	dim3 blockDim_l2(input_channels);

	input_pad << < gridDim_l2, blockDim_l2 >> > (gpu_output_m0, input_size, gpu_input_l2, pad_size);
	cudaFree(gpu_output_m0);

	// convert to columns using im2col
	double* gpu_im2col_l2;
	cudaMalloc((void**)& gpu_im2col_l2, pow(output_size ,2) * pow(kernel_size, 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I2(1);
	dim3 blockDim_I2(output_size);
	input2col << < gridDim_I2, blockDim_I2 >> > (gpu_input_l2, input_size, gpu_im2col_l2, input_channels, output_size, stride);
	cudaFree(gpu_input_l2);

	//Perform convolution for the layer

	//Read the weights and biases
	W2 = (double*)calloc(sizeof_CNN_W[2], sizeof(double));
	B2 = (double*)calloc(sizeof_CNN_B[2], sizeof(double));
	read_model_params(layer2, W2, B2, sizeof_CNN_W[2]);

	//Alocate memory for output
	double* gpu_W2, * gpu_B2, * gpu_output_l2;
	cudaMalloc((void**)& gpu_W2, sizeof_CNN_W[2] * sizeof(double));
	cudaMalloc((void**)& gpu_B2, sizeof_CNN_B[2] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l2, pow(output_size,2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W2, W2, sizeof_CNN_W[2] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B2, B2, sizeof_CNN_B[2] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W2(1);
	dim3 blockDim_W2(output_channels);
	weight_mult << < gridDim_W2, blockDim_W2 >> > (gpu_W2, gpu_B2, gpu_im2col_l2, input_channels, output_size, gpu_output_l2);
	free(W2);
	free(B2);

	cudaFree(gpu_W2);
	cudaFree(gpu_B2);
	cudaFree(gpu_im2col_l2);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-3 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 3: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 128;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l3;
	cudaMalloc((void**)& gpu_input_l3, pow((input_size + 2 * pad_size) , 2) * input_channels * sizeof(double));
	dim3 gridDim_l3(1);
	dim3 blockDim_l3(input_channels);

	input_pad << < gridDim_l3, blockDim_l3 >> > (gpu_output_l2, input_size, gpu_input_l3, pad_size);
	cudaFree(gpu_output_l2);

	// convert to columns using im2col
	double* gpu_im2col_l3;
	cudaMalloc((void**)& gpu_im2col_l3, pow(output_size,2) * pow(kernel_size, 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I3(1);
	dim3 blockDim_I3(output_size);
	input2col << < gridDim_I3, blockDim_I3 >> > (gpu_input_l3, input_size, gpu_im2col_l3, input_channels, output_size, stride);
	cudaFree(gpu_input_l3);

	//Perform convolution for the layer

	//Read the weights and biases
	W3 = (double*)calloc(sizeof_CNN_W[3], sizeof(double));
	B3 = (double*)calloc(sizeof_CNN_B[3], sizeof(double));
	read_model_params(layer3, W3, B3, sizeof_CNN_W[3]);

	//Alocate memory for output
	double* gpu_W3, * gpu_B3, * gpu_output_l3;
	cudaMalloc((void**)& gpu_W3, sizeof_CNN_W[3] * sizeof(double));
	cudaMalloc((void**)& gpu_B3, sizeof_CNN_B[3] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l3, pow(output_size , 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W3, W3, sizeof_CNN_W[3] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B3, B3, sizeof_CNN_B[3] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W3(1);
	dim3 blockDim_W3(output_channels);
	weight_mult << < gridDim_W3, blockDim_W3 >> > (gpu_W3, gpu_B3, gpu_im2col_l3, input_channels, output_size, gpu_output_l3);
	free(W3);
	free(B3);

	cudaFree(gpu_W3);
	cudaFree(gpu_B3);
	cudaFree(gpu_im2col_l3);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////// MAXPOOL-1 /////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//set the parameters
	kernel_size = 2;
	stride = 2;
	input_size = output_size;
	output_size = int(input_size / 2);
	input_channels = output_channels;
	output_channels = input_channels;
	double* gpu_output_m1;
	cudaMalloc((void**)& gpu_output_m1, pow(output_size , 2) * output_channels * sizeof(double));

	dim3 gridDim_M1(1);
	dim3 blockDim_M1(output_channels);
	maxpool << < gridDim_M1, blockDim_M1 >> > (gpu_output_l3, input_size, gpu_output_m1, output_size, stride);
	cudaFree(gpu_output_l3);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-4 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 4: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 256;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l4;
	cudaMalloc((void**)& gpu_input_l4, pow((input_size + 2 * pad_size) , 2) * input_channels * sizeof(double));
	dim3 gridDim_l4(1);
	dim3 blockDim_l4(input_channels);

	input_pad << < gridDim_l4, blockDim_l4 >> > (gpu_output_m1, input_size, gpu_input_l4, pad_size);
	cudaFree(gpu_output_m1);

	// convert to columns using im2col
	double* gpu_im2col_l4;
	cudaMalloc((void**)& gpu_im2col_l4, pow(output_size , 2) * pow(kernel_size , 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I4(1);
	dim3 blockDim_I4(output_size);
	input2col << < gridDim_I4, blockDim_I4 >> > (gpu_input_l4, input_size, gpu_im2col_l4, input_channels, output_size, stride);
	cudaFree(gpu_input_l4);

	//Perform convolution for the layer

	//Read the weights and biases
	W4 = (double*)calloc(sizeof_CNN_W[4], sizeof(double));
	B4 = (double*)calloc(sizeof_CNN_B[4], sizeof(double));
	read_model_params(layer4, W4, B4, sizeof_CNN_W[4]);

	//Alocate memory for output
	double* gpu_W4, * gpu_B4, * gpu_output_l4;
	cudaMalloc((void**)& gpu_W4, sizeof_CNN_W[4] * sizeof(double));
	cudaMalloc((void**)& gpu_B4, sizeof_CNN_B[4] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l4, pow(output_size, 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W4, W4, sizeof_CNN_W[4] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B4, B4, sizeof_CNN_B[4] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W4(1);
	dim3 blockDim_W4(output_channels);
	weight_mult << < gridDim_W4, blockDim_W4 >> > (gpu_W4, gpu_B4, gpu_im2col_l4, input_channels, output_size, gpu_output_l4);
	free(W4);
	free(B4);

	cudaFree(gpu_W4);
	cudaFree(gpu_B4);
	cudaFree(gpu_im2col_l4);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-5 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 5: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 256;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l5;
	cudaMalloc((void**)& gpu_input_l5, pow((input_size + 2 * pad_size), 2) * input_channels * sizeof(double));
	dim3 gridDim_l5(1);
	dim3 blockDim_l5(input_channels);

	input_pad << < gridDim_l5, blockDim_l5 >> > (gpu_output_l4, input_size, gpu_input_l5, pad_size);
	cudaFree(gpu_output_l4);

	// convert to columns using im2col
	double* gpu_im2col_l5;
	cudaMalloc((void**)& gpu_im2col_l5, pow(output_size, 2) * pow(kernel_size , 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I5(1);
	dim3 blockDim_I5(output_size);
	input2col << < gridDim_I5, blockDim_I5 >> > (gpu_input_l5, input_size, gpu_im2col_l5, input_channels, output_size, stride);
	cudaFree(gpu_input_l5);

	//Perform convolution for the layer

	//Read the weights and biases
	W5 = (double*)calloc(sizeof_CNN_W[5], sizeof(double));
	B5 = (double*)calloc(sizeof_CNN_B[5], sizeof(double));
	read_model_params(layer5, W5, B5, sizeof_CNN_W[5]);

	//Alocate memory for output
	double* gpu_W5, * gpu_B5, * gpu_output_l5;
	cudaMalloc((void**)& gpu_W5, sizeof_CNN_W[5] * sizeof(double));
	cudaMalloc((void**)& gpu_B5, sizeof_CNN_B[5] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l5, pow(output_size , 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W5, W5, sizeof_CNN_W[5] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B5, B5, sizeof_CNN_B[5] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W5(1);
	dim3 blockDim_W5(output_channels);
	weight_mult << < gridDim_W5, blockDim_W5 >> > (gpu_W5, gpu_B5, gpu_im2col_l5, input_channels, output_size, gpu_output_l5);
	free(W5);
	free(B5);

	cudaFree(gpu_W5);
	cudaFree(gpu_B5);
	cudaFree(gpu_im2col_l5);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-6 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 6: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 256;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l6;
	cudaMalloc((void**)& gpu_input_l6, pow((input_size + 2 * pad_size) ,2) * input_channels * sizeof(double));
	dim3 gridDim_l6(1);
	dim3 blockDim_l6(input_channels);

	input_pad << < gridDim_l6, blockDim_l6 >> > (gpu_output_l5, input_size, gpu_input_l6, pad_size);
	cudaFree(gpu_output_l5);

	// convert to columns using im2col
	double* gpu_im2col_l6;
	cudaMalloc((void**)& gpu_im2col_l6, pow(output_size , 2) * pow(kernel_size , 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I6(1);
	dim3 blockDim_I6(output_size);
	input2col << < gridDim_I6, blockDim_I6 >> > (gpu_input_l6, input_size, gpu_im2col_l6, input_channels, output_size, stride);
	cudaFree(gpu_input_l6);

	//Perform convolution for the layer

	//Read the weights and biases
	W6 = (double*)calloc(sizeof_CNN_W[6], sizeof(double));
	B6 = (double*)calloc(sizeof_CNN_B[6], sizeof(double));
	read_model_params(layer6, W6, B6, sizeof_CNN_W[6]);

	//Alocate memory for output
	double* gpu_W6, * gpu_B6, * gpu_output_l6;
	cudaMalloc((void**)& gpu_W6, sizeof_CNN_W[6] * sizeof(double));
	cudaMalloc((void**)& gpu_B6, sizeof_CNN_B[6] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l6, pow(output_size , 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W6, W6, sizeof_CNN_W[6] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B6, B6, sizeof_CNN_B[6] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W6(1);
	dim3 blockDim_W6(output_channels);
	weight_mult << < gridDim_W6, blockDim_W6 >> > (gpu_W6, gpu_B6, gpu_im2col_l6, input_channels, output_size, gpu_output_l6);
	free(W6);
	free(B6);

	cudaFree(gpu_W6);
	cudaFree(gpu_B6);
	cudaFree(gpu_im2col_l6);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-7 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 7: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 256;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l7;
	cudaMalloc((void**)& gpu_input_l7, pow((input_size + 2 * pad_size) , 2) * input_channels * sizeof(double));
	dim3 gridDim(1);
	dim3 blockDim(input_channels);

	input_pad << < gridDim, blockDim >> > (gpu_output_l6, input_size, gpu_input_l7, pad_size);
	cudaFree(gpu_output_l6);

	// convert to columns using im2col
	double* gpu_im2col_l7;
	cudaMalloc((void**)& gpu_im2col_l7, pow(output_size,2) * pow(kernel_size,2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_l7(1);
	dim3 blockDim_l7(output_size);
	input2col << < gridDim_l7, blockDim_l7 >> > (gpu_input_l7, input_size, gpu_im2col_l7, input_channels, output_size, stride);
	cudaFree(gpu_input_l7);

	//Perform convolution for the layer

	//Read the weights and biases
	W7 = (double*)calloc(sizeof_CNN_W[7], sizeof(double));
	B7 = (double*)calloc(sizeof_CNN_B[7], sizeof(double));
	read_model_params(layer7, W7, B7, sizeof_CNN_W[7]);

	//Alocate memory for output
	double* gpu_W7, * gpu_B7, * gpu_output_l7;
	cudaMalloc((void**)& gpu_W7, sizeof_CNN_W[7] * sizeof(double));
	cudaMalloc((void**)& gpu_B7, sizeof_CNN_B[7] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l7, pow(output_size,2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W7, W7, sizeof_CNN_W[7] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B7, B7, sizeof_CNN_B[7] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W7(1);
	dim3 blockDim_W7(output_channels);
	weight_mult << < gridDim_W7, blockDim_W7 >> > (gpu_W7, gpu_B7, gpu_im2col_l7, input_channels, output_size, gpu_output_l7);
	free(W7);
	free(B7);

	cudaFree(gpu_W7);
	cudaFree(gpu_B7);
	cudaFree(gpu_im2col_l7);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////// MAXPOOL-2 /////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//set the parameters

	kernel_size = 2;
	stride = 2;
	input_size = output_size;
	output_size = int(input_size / 2);
	input_channels = output_channels;
	output_channels = input_channels;
	double* gpu_output_m2;
	cudaMalloc((void**)& gpu_output_m2, pow(output_size, 2) * output_channels * sizeof(double));

	dim3 gridDim_M2(1);
	dim3 blockDim_M2(output_channels);
	maxpool << < gridDim_M2, blockDim_M2 >> > (gpu_output_l7, input_size, gpu_output_m2, output_size, stride);
	cudaFree(gpu_output_l7);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-8 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 8: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 512;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l8;
	cudaMalloc((void**)& gpu_input_l8, pow((input_size + 2 * pad_size), 2) * input_channels * sizeof(double));
	dim3 gridDim_l8(1);
	dim3 blockDim_l8(input_channels);

	input_pad << < gridDim_l8, blockDim_l8 >> > (gpu_output_m2, input_size, gpu_input_l8, pad_size);
	cudaFree(gpu_output_m2);

	// convert to columns using im2col
	double* gpu_im2col_l8;
	cudaMalloc((void**)& gpu_im2col_l8, pow(output_size , 2) * pow(kernel_size, 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I8(1);
	dim3 blockDim_I8(output_size);
	input2col << < gridDim_I8, blockDim_I8 >> > (gpu_input_l8, input_size, gpu_im2col_l8, input_channels, output_size, stride);
	cudaFree(gpu_input_l8);

	//Perform convolution for the layer

	//Read the weights and biases
	W8 = (double*)calloc(sizeof_CNN_W[8], sizeof(double));
	B8 = (double*)calloc(sizeof_CNN_B[8], sizeof(double));
	read_model_params(layer8, W8, B8, sizeof_CNN_W[8]);

	//Alocate memory for output
	double* gpu_W8, * gpu_B8, * gpu_output_l8;
	cudaMalloc((void**)& gpu_W8, sizeof_CNN_W[8] * sizeof(double));
	cudaMalloc((void**)& gpu_B8, sizeof_CNN_B[8] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l8, pow(output_size , 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W8, W8, sizeof_CNN_W[8] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B8, B8, sizeof_CNN_B[8] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W8(1);
	dim3 blockDim_W8(output_channels);
	weight_mult << < gridDim, blockDim >> > (gpu_W8, gpu_B8, gpu_im2col_l8, input_channels, output_size, gpu_output_l8);
	free(W8);
	free(B8);

	cudaFree(gpu_W8);
	cudaFree(gpu_B8);
	cudaFree(gpu_im2col_l8);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-9 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 9: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 512;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l9;
	cudaMalloc((void**)& gpu_input_l9, pow((input_size + 2 * pad_size) , 2) * input_channels * sizeof(double));
	dim3 gridDim_l9(1);
	dim3 blockDim_l9(input_channels);

	input_pad << < gridDim_l9, blockDim_l9 >> > (gpu_output_l8, input_size, gpu_input_l9, pad_size);
	cudaFree(gpu_output_l8);

	// convert to columns using im2col
	double* gpu_im2col_l9;
	cudaMalloc((void**)& gpu_im2col_l9, pow(output_size , 2) * pow(kernel_size , 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I9(1);
	dim3 blockDim_I9(output_size);
	input2col << < gridDim_I9, blockDim_I9 >> > (gpu_input_l9, input_size, gpu_im2col_l9, input_channels, output_size, stride);
	cudaFree(gpu_input_l9);

	//Perform convolution for the layer

	//Read the weights and biases
	W9 = (double*)calloc(sizeof_CNN_W[9], sizeof(double));
	B9 = (double*)calloc(sizeof_CNN_B[9], sizeof(double));
	read_model_params(layer9, W9, B9, sizeof_CNN_W[9]);

	//Alocate memory for output
	double* gpu_W9, * gpu_B9, * gpu_output_l9;
	cudaMalloc((void**)& gpu_W9, sizeof_CNN_W[9] * sizeof(double));
	cudaMalloc((void**)& gpu_B9, sizeof_CNN_B[9] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l9, pow(output_size , 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W9, W9, sizeof_CNN_W[9] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B9, B9, sizeof_CNN_B[9] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W9(1);
	dim3 blockDim_W9(output_channels);
	weight_mult << < gridDim_W9, blockDim_W9 >> > (gpu_W9, gpu_B9, gpu_im2col_l9, input_channels, output_size, gpu_output_l9);
	free(W9);
	free(B9);

	cudaFree(gpu_W9);
	cudaFree(gpu_B9);
	cudaFree(gpu_im2col_l9);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-10 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 10: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 512;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l10;
	cudaMalloc((void**)& gpu_input_l10, pow((input_size + 2 * pad_size), 2) * input_channels * sizeof(double));
	dim3 gridDim_l10(1);
	dim3 blockDim_l10(input_channels);

	input_pad << < gridDim_l10, blockDim_l10 >> > (gpu_output_l9, input_size, gpu_input_l10, pad_size);
	cudaFree(gpu_output_l9);

	// convert to columns using im2col
	double* gpu_im2col_l10;
	cudaMalloc((void**)& gpu_im2col_l10, pow(output_size ,2) * pow(kernel_size ,2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I10(1);
	dim3 blockDim_I10(output_size);
	input2col << < gridDim_I10, blockDim_I10 >> > (gpu_input_l10, input_size, gpu_im2col_l10, input_channels, output_size, stride);
	cudaFree(gpu_input_l10);

	//Perform convolution for the layer

	//Read the weights and biases
	W10 = (double*)calloc(sizeof_CNN_W[10], sizeof(double));
	B10 = (double*)calloc(sizeof_CNN_B[10], sizeof(double));
	read_model_params(layer10, W10, B10, sizeof_CNN_W[10]);

	//Alocate memory for output
	double* gpu_W10, * gpu_B10, * gpu_output_l10;
	cudaMalloc((void**)& gpu_W10, sizeof_CNN_W[10] * sizeof(double));
	cudaMalloc((void**)& gpu_B10, sizeof_CNN_B[10] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l10, pow(output_size, 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W10, W10, sizeof_CNN_W[10] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B10, B10, sizeof_CNN_B[10] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W10(1);
	dim3 blockDim_W10(output_channels);
	weight_mult << < gridDim, blockDim >> > (gpu_W10, gpu_B10, gpu_im2col_l10, input_channels, output_size, gpu_output_l10);
	free(W10);
	free(B10);

	cudaFree(gpu_W10);
	cudaFree(gpu_B10);
	cudaFree(gpu_im2col_l10);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////// LAYER-11 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 11: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 512;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l11;
	cudaMalloc((void**)& gpu_input_l11, pow((input_size + 2 * pad_size), 2) * input_channels * sizeof(double));
	dim3 gridDim_l11(1);
	dim3 blockDim_l11(input_channels);

	input_pad << < gridDim_l11, blockDim_l11 >> > (gpu_output_l10, input_size, gpu_input_l11, pad_size);
	cudaFree(gpu_output_l10);

	// convert to columns using im2col
	double* gpu_im2col_l11;
	cudaMalloc((void**)& gpu_im2col_l11, pow(output_size , 2) * pow(kernel_size , 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I11(1);
	dim3 blockDim_I11(output_size);
	input2col << < gridDim_I11, blockDim_I11 >> > (gpu_input_l11, input_size, gpu_im2col_l11, input_channels, output_size, stride);
	cudaFree(gpu_input_l11);

	//Perform convolution for the layer

	//Read the weights and biases
	W11 = (double*)calloc(sizeof_CNN_W[11], sizeof(double));
	B11 = (double*)calloc(sizeof_CNN_B[11], sizeof(double));
	read_model_params(layer11, W11, B11, sizeof_CNN_W[11]);

	//Alocate memory for output
	double* gpu_W11, * gpu_B11, * gpu_output_l11;
	cudaMalloc((void**)& gpu_W11, sizeof_CNN_W[11] * sizeof(double));
	cudaMalloc((void**)& gpu_B11, sizeof_CNN_B[11] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l11, pow(output_size ,2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W11, W11, sizeof_CNN_W[11] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B11, B11, sizeof_CNN_B[11] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W11(1);
	dim3 blockDim_W11(output_channels);
	weight_mult << < gridDim_W11, blockDim_W11 >> > (gpu_W11, gpu_B11, gpu_im2col_l11, input_channels, output_size, gpu_output_l11);
	free(W11);
	free(B11);

	cudaFree(gpu_W11);
	cudaFree(gpu_B11);
	cudaFree(gpu_im2col_l11);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////// MAXPOOL-3 /////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//set the parameters
	kernel_size = 2;
	stride = 2;
	input_size = output_size;
	output_size = int(input_size / 2);
	input_channels = output_channels;
	output_channels = input_channels;
	double* gpu_output_m3;
	cudaMalloc((void**)& gpu_output_m3, pow(output_size , 2) * output_channels * sizeof(double));

	dim3 gridDim_M3(1);
	dim3 blockDim_M3(output_channels);
	maxpool << < gridDim_M3, blockDim_M3 >> > (gpu_output_l11, input_size, gpu_output_m3, output_size, stride);
	cudaFree(gpu_output_l11);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-12 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 12: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 512;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l12;
	cudaMalloc((void**)& gpu_input_l12, pow((input_size + 2 * pad_size) , 2) * input_channels * sizeof(double));
	dim3 gridDim_l12(1);
	dim3 blockDim_l12(input_channels);

	input_pad << < gridDim_l12, blockDim_l12 >> > (gpu_output_m3, input_size, gpu_input_l12, pad_size);
	cudaFree(gpu_output_m3);

	// convert to columns using im2col
	double* gpu_im2col_l12;
	cudaMalloc((void**)& gpu_im2col_l12, pow(output_size ,2) * pow(kernel_size , 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I12(1);
	dim3 blockDim_I12(output_size);
	input2col << < gridDim_I12, blockDim_I12 >> > (gpu_input_l12, input_size, gpu_im2col_l12, input_channels, output_size, stride);
	cudaFree(gpu_input_l12);

	//Perform convolution for the layer

	//Read the weights and biases
	W12 = (double*)calloc(sizeof_CNN_W[12], sizeof(double));
	B12 = (double*)calloc(sizeof_CNN_B[12], sizeof(double));
	read_model_params(layer12, W12, B12, sizeof_CNN_W[12]);

	//Alocate memory for output
	double* gpu_W12, * gpu_B12, * gpu_output_l12;
	cudaMalloc((void**)& gpu_W12, sizeof_CNN_W[12] * sizeof(double));
	cudaMalloc((void**)& gpu_B12, sizeof_CNN_B[12] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l12, pow(output_size , 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W12, W12, sizeof_CNN_W[12] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B12, B12, sizeof_CNN_B[12] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W12(1);
	dim3 blockDim_W12(output_channels);
	weight_mult << < gridDim_W12, blockDim_W12 >> > (gpu_W12, gpu_B12, gpu_im2col_l12, input_channels, output_size, gpu_output_l12);
	free(W12);
	free(B12);

	cudaFree(gpu_W12);
	cudaFree(gpu_B12);
	cudaFree(gpu_im2col_l12);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-13 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 13: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 512;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l13;
	cudaMalloc((void**)& gpu_input_l13, pow((input_size + 2 * pad_size) , 2) * input_channels * sizeof(double));
	dim3 gridDim_l13(1);
	dim3 blockDim_l13(input_channels);

	input_pad << < gridDim_l13, blockDim_l13 >> > (gpu_output_l12, input_size, gpu_input_l13, pad_size);
	cudaFree(gpu_output_l12);

	// convert to columns using im2col
	double* gpu_im2col_l13;
	cudaMalloc((void**)& gpu_im2col_l13, pow(output_size , 2) * pow(kernel_size , 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I13(1);
	dim3 blockDim_I13(output_size);
	input2col << < gridDim_I13, blockDim_I13 >> > (gpu_input_l13, input_size, gpu_im2col_l13, input_channels, output_size, stride);
	cudaFree(gpu_input_l13);

	//Perform convolution for the layer

	//Read the weights and biases
	W13 = (double*)calloc(sizeof_CNN_W[13], sizeof(double));
	B13 = (double*)calloc(sizeof_CNN_B[13], sizeof(double));
	read_model_params(layer13, W13, B13, sizeof_CNN_W[13]);

	//Alocate memory for output
	double* gpu_W13, * gpu_B13, * gpu_output_l13;
	cudaMalloc((void**)& gpu_W13, sizeof_CNN_W[13] * sizeof(double));
	cudaMalloc((void**)& gpu_B13, sizeof_CNN_B[13] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l13, pow(output_size , 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W13, W13, sizeof_CNN_W[13] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B13, B13, sizeof_CNN_B[13] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W13(1);
	dim3 blockDim_W13(output_channels);
	weight_mult << < gridDim_W13, blockDim_W13 >> > (gpu_W13, gpu_B13, gpu_im2col_l13, input_channels, output_size, gpu_output_l13);
	free(W13);
	free(B13);

	cudaFree(gpu_W13);
	cudaFree(gpu_B13);
	cudaFree(gpu_im2col_l13);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////// LAYER-14 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 14: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 512;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l14;
	cudaMalloc((void**)& gpu_input_l14, pow((input_size + 2 * pad_size) , 2) * input_channels * sizeof(double));
	dim3 gridDim_l14(1);
	dim3 blockDim_l14(input_channels);

	input_pad << < gridDim_l14, blockDim_l14 >> > (gpu_output_l13, input_size, gpu_input_l14, pad_size);
	cudaFree(gpu_output_l13);

	// convert to columns using im2col
	double* gpu_im2col_l14;
	cudaMalloc((void**)& gpu_im2col_l14, pow(output_size , 2) * pow(kernel_size , 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I14(1);
	dim3 blockDim_I14(output_size);
	input2col << < gridDim_I14, blockDim_I14 >> > (gpu_input_l14, input_size, gpu_im2col_l14, input_channels, output_size, stride);
	cudaFree(gpu_input_l14);

	//Perform convolution for the layer

	//Read the weights and biases
	W14 = (double*)calloc(sizeof_CNN_W[14], sizeof(double));
	B14 = (double*)calloc(sizeof_CNN_B[14], sizeof(double));
	read_model_params(layer14, W14, B14, sizeof_CNN_W[14]);

	//Alocate memory for output
	double* gpu_W14, * gpu_B14, * gpu_output_l14;
	cudaMalloc((void**)& gpu_W14, sizeof_CNN_W[14] * sizeof(double));
	cudaMalloc((void**)& gpu_B14, sizeof_CNN_B[14] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l14, pow(output_size , 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W14, W14, sizeof_CNN_W[14] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B14, B14, sizeof_CNN_B[14] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W14(1);
	dim3 blockDim_W14(output_channels);
	weight_mult << < gridDim_W14, blockDim_W14 >> > (gpu_W14, gpu_B14, gpu_im2col_l14, input_channels, output_size, gpu_output_l14);
	free(W14);
	free(B14);

	cudaFree(gpu_W14);
	cudaFree(gpu_B14);
	cudaFree(gpu_im2col_l14);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////// LAYER-15 //////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///Set the parameters: no of channels, stride, kernel size, output image size
	my_time = time(NULL);
	cout << "Executing Layer 15: " << ctime(&my_time) << endl;
	kernel_size = 3;
	stride = 1;
	input_channels = output_channels;
	input_size = output_size;
	output_channels = 512;
	pad_size = 1;
	output_size = floor((input_size - kernel_size + 2 * pad_size) / stride) + 1;

	// pad the input
	double* gpu_input_l15;
	cudaMalloc((void**)& gpu_input_l15, pow((input_size + 2 * pad_size) , 2) * input_channels * sizeof(double));
	dim3 gridDim_l15(1);
	dim3 blockDim_l15(input_channels);

	input_pad << < gridDim_l15, blockDim_l15 >> > (gpu_output_l14, input_size, gpu_input_l15, pad_size);
	cudaFree(gpu_output_l14);

	// convert to columns using im2col
	double* gpu_im2col_l15;
	cudaMalloc((void**)& gpu_im2col_l15, pow(output_size , 2) * pow(kernel_size , 2) * input_channels * sizeof(double));
	input_size = input_size + 2 * pad_size;
	dim3 gridDim_I15(1);
	dim3 blockDim_I15(output_size);
	input2col << < gridDim_I15, blockDim_I15 >> > (gpu_input_l15, input_size, gpu_im2col_l15, input_channels, output_size, stride);
	cudaFree(gpu_input_l15);

	//Perform convolution for the layer

	//Read the weights and biases
	W15 = (double*)calloc(sizeof_CNN_W[15], sizeof(double));
	B15 = (double*)calloc(sizeof_CNN_B[15], sizeof(double));
	read_model_params(layer15, W15, B15, sizeof_CNN_W[15]);

	//Alocate memory for output
	double* gpu_W15, * gpu_B15, * gpu_output_l15;
	cudaMalloc((void**)& gpu_W15, sizeof_CNN_W[15] * sizeof(double));
	cudaMalloc((void**)& gpu_B15, sizeof_CNN_B[15] * sizeof(double));
	cudaMalloc((void**)& gpu_output_l15, pow(output_size , 2) * output_channels * sizeof(double));
	cudaMemcpy(gpu_W15, W15, sizeof_CNN_W[15] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B15, B15, sizeof_CNN_B[15] * sizeof(double), cudaMemcpyHostToDevice);

	dim3 gridDim_W15(1);
	dim3 blockDim_W15(output_channels);
	weight_mult << < gridDim_W15, blockDim_W15 >> > (gpu_W15, gpu_B15, gpu_im2col_l15, input_channels, output_size, gpu_output_l15);
	free(W15);
	free(B15);

	cudaFree(gpu_W15);
	cudaFree(gpu_B15);
	cudaFree(gpu_im2col_l15);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////// MAXPOOL-4 /////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//set the parameters
	kernel_size = 2;
	stride = 2;
	input_size = output_size;
	output_size = int(input_size / 2);
	input_channels = output_channels;
	output_channels = input_channels;
	double* gpu_output_m4;
	cudaMalloc((void**)& gpu_output_m4, pow(output_size , 2) * output_channels * sizeof(double));
	double* cpu_m4 = (double*)calloc(pow(output_size , 2) * output_channels, sizeof(double));
	dim3 gridDim_M4(1);
	dim3 blockDim_M4(output_channels);
	maxpool << < gridDim_M4, blockDim_M4 >> > (gpu_output_l15, input_size, gpu_output_m4, output_size, stride);
	cudaFree(gpu_output_l15);
	cudaMemcpy(cpu_m4, gpu_output_m4, pow(output_size , 2) * output_channels * sizeof(double), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 100; i++)
		//cout << cpu_m4[i] << endl;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////// FC-0 //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	my_time = time(NULL);
	cout << "Executing Fully Connected Layer 0: " << ctime(&my_time) << endl;
	input_size = pow(output_size , 2) * output_channels;
	output_size = 4096;

	double* FC0_W = (double*)calloc(sizeof_FC_W[0], sizeof(double));
	double* FC0_B = (double*)calloc(sizeof_FC_B[0], sizeof(double));
	read_model_params(fclayer0, FC0_W, FC0_B, sizeof_FC_W[0]);

	double* gpu_FC0_W, * gpu_FC0_B, * gpu_output_FC0;
	cudaMalloc((void**)& gpu_FC0_W, sizeof_FC_W[0] * sizeof(double));
	cudaMalloc((void**)& gpu_FC0_B, sizeof_FC_B[0] * sizeof(double));
	cudaMalloc((void**)& gpu_output_FC0, output_size * sizeof(double));
	cudaMemcpy(gpu_FC0_W, FC0_W, sizeof_FC_W[0] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_FC0_B, FC0_B, sizeof_FC_B[0] * sizeof(double), cudaMemcpyHostToDevice);
	dim3 gridDim_F0(1);
	dim3 blockDim_F0(int(output_size / 4));
	fc_mult << < gridDim_F0, blockDim_F0 >> > (gpu_output_m4, gpu_FC0_W, gpu_FC0_B, input_size, output_size, gpu_output_FC0);

	free(FC0_W);
	free(FC0_B);

	cudaFree(gpu_FC0_W);
	cudaFree(gpu_FC0_B);
	cudaFree(gpu_output_m4);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////// FC-1 //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	my_time = time(NULL);
	cout << "Executing Fully Connected Layer 1: " << ctime(&my_time) << endl;
	input_size = output_size;
	output_size = 4096;

	double* FC1_W = (double*)calloc(sizeof_FC_W[1], sizeof(double));
	double* FC1_B = (double*)calloc(sizeof_FC_B[1], sizeof(double));
	read_model_params(fclayer1, FC1_W, FC1_B, sizeof_FC_W[1]);

	double* gpu_FC1_W, * gpu_FC1_B, * gpu_output_FC1;
	cudaMalloc((void**)& gpu_FC1_W, sizeof_FC_W[1] * sizeof(double));
	cudaMalloc((void**)& gpu_FC1_B, sizeof_FC_B[1] * sizeof(double));
	cudaMalloc((void**)& gpu_output_FC1, output_size * sizeof(double));
	cudaMemcpy(gpu_FC1_W, FC1_W, sizeof_FC_W[1] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_FC1_B, FC1_B, sizeof_FC_B[1] * sizeof(double), cudaMemcpyHostToDevice);
	dim3 gridDim_F1(1);
	dim3 blockDim_F1(int(output_size / 4));
	fc_mult << < gridDim_F1, blockDim_F1 >> > (gpu_output_FC0, gpu_FC1_W, gpu_FC1_B, input_size, output_size, gpu_output_FC1);

	free(FC1_W);
	free(FC1_B);

	cudaFree(gpu_FC1_W);
	cudaFree(gpu_FC1_B);
	cudaFree(gpu_output_FC0);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////// FC-2 //////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	my_time = time(NULL);
	cout << "Executing Fully Connected Layer 2: " << ctime(&my_time) << endl;
	input_size = output_size;
	output_size = 1000;

	double* FC2_W = (double*)calloc(sizeof_FC_W[2], sizeof(double));
	double* FC2_B = (double*)calloc(sizeof_FC_B[2], sizeof(double));
	read_model_params(fclayer2, FC2_W, FC2_B, sizeof_FC_W[2]);

	double* gpu_FC2_W, * gpu_FC2_B, * gpu_output_FC2;
	cudaMalloc((void**)& gpu_FC2_W, sizeof_FC_W[2] * sizeof(double));
	cudaMalloc((void**)& gpu_FC2_B, sizeof_FC_B[2] * sizeof(double));
	cudaMalloc((void**)& gpu_output_FC2, output_size * sizeof(double));
	cudaMemcpy(gpu_FC2_W, FC2_W, sizeof_FC_W[2] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_FC2_B, FC2_B, sizeof_FC_B[2] * sizeof(double), cudaMemcpyHostToDevice);
	dim3 gridDim_F2(1);
	dim3 blockDim_F2(output_size);
	fc_mult << < gridDim_F2, blockDim_F2 >> > (gpu_output_FC1, gpu_FC2_W, gpu_FC2_B, input_size, output_size, gpu_output_FC2);

	free(FC2_W);
	free(FC2_B);

	cudaFree(gpu_FC2_W);
	cudaFree(gpu_FC2_B);
	cudaFree(gpu_output_FC1);

	double* final_predictions = (double*)calloc(output_size, sizeof(double));
	cudaMemcpy(final_predictions, gpu_output_FC2, output_size * sizeof(double), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 100; i++)
		//cout << final_predictions[i] << endl;
	int correct_class = distance(final_predictions, max_element(final_predictions, final_predictions + output_size));
	my_time = time(NULL);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "the correct class is " << correct_class << "\t " << ctime(&my_time) << endl;

	return 0;
}
