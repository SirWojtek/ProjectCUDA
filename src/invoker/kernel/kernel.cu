
#include "kernel.cuh"
#include "gpuErrchk.cuh"

#include <assert.h>
#include <iostream>
#include <stdio.h>

__device__ int getGlobalIdx_2D_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	return threadId;
}

__device__ float sumMatrix(const float * const d_inputMatrix1, const float * const d_inputMatrix2, int index)
{
	return d_inputMatrix1[index] + d_inputMatrix2[index];
}

__device__ void injectError(float * const d_inputMatrix, const float * const d_errorMap, int index)
{
	d_inputMatrix[index] += d_errorMap[index];
}

__global__ void matrixOperation(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix)
{
	int index = getGlobalIdx_2D_2D();
	d_outputMatrix[index] = sumMatrix(d_inputMatrix1, d_inputMatrix2, index);
}

__global__ void matrixOperation(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix, float * const d_errorMap)
{
	int index = getGlobalIdx_2D_2D();
	d_outputMatrix[index] = sumMatrix(d_inputMatrix1, d_inputMatrix2, index);
	injectError(d_outputMatrix, d_errorMap, getGlobalIdx_2D_2D());
}

void fillErrorMap(float * const errorMap, const int numRows, const int numCols)
{
	for (int i = 0; i < numRows*numCols; i++)
	{
		errorMap[i] = 10;
	}
}

void runKernel(dim3 gridSize, dim3 blockSize, float* in1, float* in2, float* out)
{
	//std::cout << gridSize.x << gridSize.y << gridSize.z << "\n";
	//std::cout << blockSize.x << blockSize.y << blockSize.z << "\n";
	cudaEvent_t start, stop; // Mam pewne obawy przed wyrzucaniem tego do osobnych funkcji, �eby nie zajmowa�o niepotrzebnie czasu systemowego
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	matrixOperation <<< gridSize, blockSize >>>(in1, in2, out);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	std::cout << "Time for the Kernel: " << time << std::endl;
}

void runKernelWithError(dim3 gridSize, dim3 blockSize, float* in1, float* in2,
	float* out, float* error)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	matrixOperation <<< gridSize, blockSize >>>(in1, in2, out, error);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	std::cout << "Time for the ErrorKernel: " << time << std::endl;
}
