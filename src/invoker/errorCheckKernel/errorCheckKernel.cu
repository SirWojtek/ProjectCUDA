#include "errorCheckKernel.cuh"
#include "..\kernelCommon\gpuErrchk.cuh"

#include "..\..\matrix_loader\matrix.hpp"
#include <iostream>

__device__ int getIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__device__ int getIdx_2D_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	return threadId;
}

__device__ bool compareValues(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2)
{
	return (*d_inputMatrix1 - *d_inputMatrix2 > 0.001);
}

__global__ void errorCheckKernel(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, bool * const d_outputMatrix)
{
	int index = getIdx_1D_1D();
	d_outputMatrix[index] = compareValues(&d_inputMatrix1[index], &d_inputMatrix2[index]);
}

void runErrorCheckKernel(dim3 gridSize, dim3 blockSize, const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, bool * const d_outputMatrix)
{
	errorCheckKernel << < gridSize, blockSize >> > (d_inputMatrix1, d_inputMatrix2, d_outputMatrix);
}

void CellInfoToFloatE(float * output, CellInfo * input, int arraySize)
{
	for (int i = 0; i < arraySize; i++)
	{
		output[i] = input[i].value;
	}
}

void testErrorCheck()
{
	Matrix m1("matrixes/bcsstk03.mtx");
	Matrix m2("matrixes/bcsstk03.mtx");

	int arraySize = m1.getNonZeroValuesAmount();
	int arrayBytes = arraySize * sizeof(float);
	int arrayBytesBool = arraySize * sizeof(bool);

	// init CPU vars // no smart pointers in .cu allowed, watch out
	float *host_mIn1 = new float[arraySize];  CellInfoToFloatE(host_mIn1, m1.getMatrix(), m1.getNonZeroValuesAmount());
	float *host_mIn2 = new float[arraySize];  CellInfoToFloatE(host_mIn2, m2.getMatrix(), m2.getNonZeroValuesAmount());
	bool *host_mOut = new bool[arraySize];

	// init GPU vars
	float *device_mIn1;
	float *device_mIn2;
	bool *device_mOut;

	// alloc GPU memory
	gpuErrchk(cudaMalloc((void**)&device_mIn1, arrayBytes));
	gpuErrchk(cudaMalloc((void**)&device_mIn2, arrayBytes));
	gpuErrchk(cudaMalloc((void**)&device_mOut, arrayBytesBool));
	
	
	// copy memory to device
	gpuErrchk(cudaMemcpy(device_mIn1, host_mIn1, arrayBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(device_mIn2, host_mIn2, arrayBytes, cudaMemcpyHostToDevice));

	// launch kernel
	//kernel << <arraySize, 1 >> >(device_mIn1, device_mIn2, device_mOut);
	// launch kernel with error
	errorCheckKernel << <arraySize, 1 >> >(device_mIn1, device_mIn2, device_mOut);
	gpuErrchk(cudaPeekAtLastError()); // debug

	// copy memory from device
	gpuErrchk(cudaMemcpy(host_mOut, device_mOut, arrayBytesBool, cudaMemcpyDeviceToHost));

	std::cout << "INPUT: \n";
	for (int i = 0; i < arraySize; i++)
		std::cout << host_mIn1[i] << " ";
	std::cout << "\nOUTPUT: \n";
	for (int i = 0; i < arraySize; i++)
		std::cout << host_mOut[i] << " ";

	// cleaning
	cudaFree(device_mIn1);
	cudaFree(device_mIn2);
	cudaFree(device_mOut);
	delete[] host_mOut;
}
