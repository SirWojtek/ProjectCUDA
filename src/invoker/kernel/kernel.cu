
#include "kernel.cuh"
#include "gpuErrchk.cuh"
#include "..\..\matrix_loader\matrix.hpp"
#include <assert.h>
#include <iostream>
#include <stdio.h>

__device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_2D_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	return threadId;
}

__device__ CellInfo sumMatrix(const CellInfo * const d_inputMatrix1, const CellInfo * const d_inputMatrix2, int index)
{
	CellInfo result = {
		d_inputMatrix1[index].value + d_inputMatrix2[index].value,
		d_inputMatrix1[index].row,
		d_inputMatrix1[index].column
						};
	return result;
}

__device__ bool isError(int index)
{
	// what is the reason for error? 
	return true;
}

__device__ void injectError(CellInfo &inputCell)
{
	inputCell.value += 99999999;
}

__global__ void kernel(const CellInfo  * const d_inputMatrix1,
	const CellInfo  * const d_inputMatrix2, CellInfo * const d_outputMatrix)
{
	int index = getGlobalIdx_1D_1D();
	d_outputMatrix[index] = sumMatrix(d_inputMatrix1, d_inputMatrix2, index);
}

__global__ void kernelPlusError(const CellInfo  * const d_inputMatrix1,
	const CellInfo  * const d_inputMatrix2, CellInfo * const d_outputMatrix)
{
	int index = getGlobalIdx_1D_1D();
	d_outputMatrix[index] = sumMatrix(d_inputMatrix1, d_inputMatrix2, index);
	if (isError(index))
		injectError(d_outputMatrix[index]);
}



//
//void runKernel(dim3 gridSize, dim3 blockSize, float* in1, float* in2, float* out)
//{
//	cudaEvent_t start, stop; // Mam pewne obawy przed wyrzucaniem tego do osobnych funkcji, ¿eby nie zajmowa³o niepotrzebnie czasu systemowego
//	float time;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	cudaEventRecord(start, 0);
//
//	matrixOperation <<< gridSize, blockSize >>>(in1, in2, out);
//
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//
//	cudaEventElapsedTime(&time, start, stop);
//	std::cout << "Time for the Kernel: " << time << std::endl;
//}
//
//void runKernelWithError(dim3 gridSize, dim3 blockSize, float* in1, float* in2,
//	float* out, float* error)
//{
//	cudaEvent_t start, stop;
//	float time;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	cudaEventRecord(start, 0);
//
//	matrixOperation <<< gridSize, blockSize >>>(in1, in2, out, error);
//
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//
//	cudaEventElapsedTime(&time, start, stop);
//	std::cout << "Time for the ErrorKernel: " << time << std::endl;
//}

void testStartKernel()
{
	Matrix m1("matrixes/bcsstk03.mtx");
	Matrix m2("matrixes/bcsstk03.mtx");

	int arraySize = m1.getNonZeroValuesAmount();
	int arrayBytes = arraySize * sizeof(CellInfo);

	// init CPU vars // no smart pointers in .cu allowed, watch out
	CellInfo *host_mIn1 = new CellInfo[arraySize]; host_mIn1 = m1.getMatrix();
	CellInfo *host_mIn2 = new CellInfo[arraySize]; host_mIn2 = m2.getMatrix();
	CellInfo *host_mOut = new CellInfo[arraySize];

	// init GPU vars
	CellInfo *device_mIn1;
	CellInfo *device_mIn2;
	CellInfo *device_mOut;

	// alloc GPU memory
	gpuErrchk(cudaMalloc((void**)&device_mIn1, arrayBytes));
	gpuErrchk(cudaMalloc((void**)&device_mIn2, arrayBytes));
	gpuErrchk(cudaMalloc((void**)&device_mOut, arrayBytes));
	// copy memory to device
	gpuErrchk(cudaMemcpy(device_mIn1, host_mIn1, arrayBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(device_mIn2, host_mIn2, arrayBytes, cudaMemcpyHostToDevice));

	// launch kernel
	kernel << <arraySize, 1 >> >(device_mIn1, device_mIn2, device_mOut);
	// launch kernel with error
	//kernelPlusError << <arraySize, 1 >> >(device_mIn1, device_mIn2, device_mOut);
	gpuErrchk(cudaPeekAtLastError()); // debug

	// copy memory from device
	gpuErrchk(cudaMemcpy(host_mOut, device_mOut, arrayBytes, cudaMemcpyDeviceToHost));
	
	std::cout << "INPUT: \n";
	for (int i = 0; i < arraySize; i++)
		std::cout << host_mIn1[i].value << " ";
	std::cout << "\nOUTPUT: \n";
	for (int i = 0; i < arraySize; i++)
		std::cout << host_mOut[i].value<< " ";

	// cleaning
	cudaFree(device_mIn1);
	cudaFree(device_mIn2);
	cudaFree(device_mOut);
	delete[] host_mOut;
}