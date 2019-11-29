
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<opencv2/opencv.hpp>
#define input_file "cat.jpg"
using namespace std;
using namespace cv;
#define h 224
#define w 224
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///Kernel function to convert from image to column
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void im2col(int* r, int *g, int *b, int* result, int cols)
{
	int k;
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	k = int(i * 27 * (cols-2));
	//printf("%d\n", result[k]);
	for (int j = 1; j < cols - 1; j++)
	{
		for (int m = -1; m < 2; m++)
		{
			result[k] = r[cols * (i + 1 + m) + j - 1];
			result[k + 1] = r[cols * (i + 1 + m) + j];
			result[k + 2] = r[cols * (i + 1 + m) + j + 1];
			k = k + 3;
		}
		for (int m = -1; m < 2; m++)
		{
			result[k] = g[cols * (i + 1 + m) + j - 1];
			result[k + 1] = g[cols * (i + 1 + m) + j];
			result[k + 2] = g[cols * (i + 1 + m) + j + 1];
			k = k + 3;
		}
		for (int m = -1; m < 2; m++)
		{
			result[k] = b[cols * (i + 1 + m) + j - 1];
			result[k + 1] = b[cols * (i + 1 + m) + j];
			result[k + 2] = b[cols * (i + 1 + m) + j + 1];
			k = k + 3;
		}
	}
	
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///Function to read Dimension
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int readdim(double* dim )
{
	FILE* fp;
	ifstream myReadFile;
	myReadFile.open(dimensions);
	if (!myReadFile) {
		cout << "Unable to open file";
		exit(1); // terminate with error
	}
	int k = 0;
	if (myReadFile.is_open()) {
		while (!myReadFile.eof()) {
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
	if (!myReadFile) {
		cout << "Unable to open file";
		exit(1); // terminate with error
	}
	int k;
	if (myReadFile.is_open()) 
	{
		k = 0;
		while (!myReadFile.eof() && k < sizeofW) {
			myReadFile >> W[k];
			k++;
		}
		k = 0;
		while (!myReadFile.eof()) {
			myReadFile >> B[k];
			k++;
		}
	}
	myReadFile.close();
}

int main()
{
	///Read Dimensions FIle
	double* dim= (double*)calloc(4*16+2*3, sizeof(double));
	readdim(dim);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Set the dimensions to the Weights and Biases
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int* sizeof_CNN_W = (int*)calloc(16, sizeof(int));
	int* sizeof_CNN_B = (int*)calloc(16, sizeof(int));
	int* sizeof_FC_W= (int*)calloc(3, sizeof(int));
	int* sizeof_FC_B = (int*)calloc(3, sizeof(int));
	int k = 0;
	for (int i = 0; i < 16; i++)
	{
		sizeof_CNN_W[i] = (int)(dim[k] * dim[k+1] * dim[k+2] * dim[k+3]);
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
	/// Read the Weights and Biases of Layer 0
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	double* W0, * W1, * W2, * W3, * W4, * W5, * W6, * W7, * W8, * W9, * W10, * W11, * W12, * W13, * W14, * W15 ;
	double* B0, * B1, * B2, * B3, * B4, * B5, * B6, * B7, * B8, * B9, * B10, * B11, * B12, * B13, * B14, * B15;
	W0 = (double*)calloc(sizeof_CNN_W[0], sizeof(double));
	B0 = (double*)calloc(sizeof_CNN_B[0], sizeof(double));
	read_model_params(layer0,W0,B0,sizeof_CNN_W[0]);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*	for (int i = 0; i < sizeof_CNN_W[0]; i++)
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
	//Reading the image and convert to multiple channels R,G,B
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///Set the paramenters: no of channels, stride, kernel size, output image size
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int no_of_channels = 3;
	int stride = 1;
	int kernel_size = 3;
	int out_size = floor((rows - kernel_size) / stride) + 1;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Set the CUDA params, Allocate the GPU memory and Copy the data from CPU to GPU
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	dim3 gridDim(1);
	dim3 blockDim(rows-2);
	int* result = (int*)calloc(out_size * out_size * kernel_size * kernel_size * no_of_channels, sizeof(int));
	int* d_r, * d_g, * d_b, * d_cols, * d_result;
	cudaMalloc((void**)& d_r, rows * cols * sizeof(int));
	cudaMalloc((void**)& d_g, rows * cols * sizeof(int));
	cudaMalloc((void**)& d_b, rows * cols * sizeof(int));
	cudaMalloc((void**)& d_result, out_size * out_size * kernel_size * kernel_size * no_of_channels*sizeof(int));
	cudaMemcpy(d_r, r, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_g, g, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, rows * cols * sizeof(int), cudaMemcpyHostToDevice);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call the kernel function
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	im2col <<<gridDim, blockDim >> > (d_r,d_g,d_b,d_result,cols);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//copy from GPU to CPU
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaMemcpy(result, d_result, out_size * out_size * kernel_size * kernel_size * no_of_channels * sizeof(int), cudaMemcpyDeviceToHost);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	for (int i=0; i < 100; i++)
	{
		cout << result[i]<<endl;
	}
	free(r);
	free(g);
	free(b);
	cudaFree(d_r);
	cudaFree(d_g);
	cudaFree(d_b);
	
	return 0;
}
