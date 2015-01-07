#include "kernel.cuh"
#include "..\kernelCommon\gpuErrchk.cuh"
#include "..\..\matrix_loader\matrix.hpp"
#include <assert.h>
#include <iostream>
#include <stdio.h>


// broken kernel
__device__ int brokenBlock = 0;
__device__ int brokenThread = 15;

__device__ int getErrorIdx_1D_1D()
{
	return brokenBlock *blockDim.x + brokenThread;
}

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

__device__ float sumMatrix(const float * const d_inputMatrix1, const float * const d_inputMatrix2, int index)
{
	float result = d_inputMatrix1[index] + d_inputMatrix2[index];
	return result;
}

__device__ bool isError(int index)
{
	if (index == getErrorIdx_1D_1D())
		return true;
	else
		return false;
}

__device__ void injectError(CellInfo &inputCell)
{
	inputCell.value += 99999999;
}

__device__ void injectError(float *inputCell)
{
	*inputCell += 99999999;
}

__global__ void kernel(const CellInfo  * const d_inputMatrix1,
	const CellInfo  * const d_inputMatrix2, CellInfo * const d_outputMatrix)
{
	int index = getGlobalIdx_1D_1D();
	d_outputMatrix[index] = sumMatrix(d_inputMatrix1, d_inputMatrix2, index);
}

__global__ void kernel(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix)
{
	int index = getGlobalIdx_1D_1D();
	d_outputMatrix[index] = sumMatrix(d_inputMatrix1, d_inputMatrix2, index);
}

__global__ void kernelWithCounter(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix, int *current_thread_count)
{
	atomicAdd(current_thread_count, 1);
	int index = getGlobalIdx_1D_1D();
	d_outputMatrix[index] = sumMatrix(d_inputMatrix1, d_inputMatrix2, index);
}

void runKernel(dim3 gridSize, dim3 blockSize, const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix)
{
	kernel << < gridSize, blockSize >> > (d_inputMatrix1, d_inputMatrix2, d_outputMatrix);
}

__global__ void kernelPlusError(const CellInfo  * const d_inputMatrix1,
	const CellInfo  * const d_inputMatrix2, CellInfo * const d_outputMatrix)
{
	int index = getGlobalIdx_1D_1D();
	d_outputMatrix[index] = sumMatrix(d_inputMatrix1, d_inputMatrix2, index);
	if (isError(index))
		injectError(d_outputMatrix[index]);
}

__global__ void kernelPlusError(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix)
{
	int index = getGlobalIdx_1D_1D();
	d_outputMatrix[index] = sumMatrix(d_inputMatrix1, d_inputMatrix2, index);
	if (isError(index))
		injectError(&d_outputMatrix[index]);
}

void runKernelPlusError(dim3 gridSize, dim3 blockSize, const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix, 
	int arrayBytes, float * d_hostMatrix1, float * d_hostMatrix2,
	cudaStream_t * stream)
{
	gpuErrchk(cudaMemcpyAsync((void**)d_inputMatrix1, d_hostMatrix1, arrayBytes, cudaMemcpyHostToDevice, *stream));
	gpuErrchk(cudaMemcpyAsync((void**)d_inputMatrix2, d_hostMatrix2, arrayBytes, cudaMemcpyHostToDevice, *stream));

	kernelPlusError <<< gridSize, blockSize, 0, *stream >>> (d_inputMatrix1, d_inputMatrix2, d_outputMatrix);
}

void runCommandCenter(dim3 gridSize, dim3 blockSize, const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix, float * const d_outputMatrix2,  
	int arrayBytes, float * d_hostMatrix1, float * d_hostMatrix2)
{
	cudaStream_t stream[2];
	float * redundantMatrix1;
	float * redundantMatrix2;
	gpuErrchk(cudaMalloc((void**)&redundantMatrix1, arrayBytes));
	gpuErrchk(cudaMalloc((void**)&redundantMatrix2, arrayBytes));

	cudaEvent_t start[2], stop[2];
	float timer[2];

	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);

	cudaEventCreate(&start[0]);
  	cudaEventRecord(start[0], stream[0]);

	gpuErrchk(cudaMemcpyAsync((void**)d_inputMatrix1, d_hostMatrix1, arrayBytes, cudaMemcpyHostToDevice, stream[0]));
	gpuErrchk(cudaMemcpyAsync((void**)d_inputMatrix2, d_hostMatrix2, arrayBytes, cudaMemcpyHostToDevice, stream[0]));
	kernelPlusError <<< gridSize, blockSize, 0, stream[0] >>> (d_inputMatrix1, d_inputMatrix2, d_outputMatrix);

	cudaEventCreate(&stop[0]);
	cudaEventRecord(stop[0],stream[0]);
	cudaEventSynchronize(stop[0]);

	cudaEventCreate(&start[1]);
  	cudaEventRecord(start[1], stream[1]);

	gpuErrchk(cudaMemcpyAsync((void**)redundantMatrix1, d_hostMatrix1, arrayBytes, cudaMemcpyHostToDevice, stream[1]));
	gpuErrchk(cudaMemcpyAsync((void**)redundantMatrix2, d_hostMatrix2, arrayBytes, cudaMemcpyHostToDevice, stream[1]));
	kernel <<< gridSize, blockSize, 0, stream[1] >>> (redundantMatrix1, redundantMatrix2, d_outputMatrix2);	


	cudaEventCreate(&stop[1]);
	cudaEventRecord(stop[1],stream[1]);
	cudaEventSynchronize(stop[1]);

	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
	cudaEventElapsedTime(&timer[0], start[0],stop[0]);
	cudaEventElapsedTime(&timer[1], start[1],stop[1]);
	std::cout << "Error calculation time [ms]: " << timer[0] << std::endl;
	std::cout << "Redundant calculation time [ms]: " << timer[1] << std::endl;

	cudaFree(redundantMatrix1);
	cudaFree(redundantMatrix2);
}

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

void CellInfoToFloat(float * output, CellInfo * input, int arraySize)
{
	for (int i = 0; i < arraySize; i++)
	{
		output[i] = input[i].value;
	}
}

void testStartKernel_CellInfo()
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

void testStartKernel_float()
{
	Matrix m1("matrixes/bcsstk03.mtx");
	Matrix m2("matrixes/bcsstk03.mtx");

	int arraySize = m1.getNonZeroValuesAmount();
	int arrayBytes = arraySize * sizeof(float);

	// init CPU vars // no smart pointers in .cu allowed, watch out
	float *host_mIn1 = new float[arraySize];  CellInfoToFloat(host_mIn1, m1.getMatrix(), m1.getNonZeroValuesAmount());
	float *host_mIn2 = new float[arraySize];  CellInfoToFloat(host_mIn2, m2.getMatrix(), m2.getNonZeroValuesAmount());
	float *host_mOut = new float[arraySize];

	// init GPU vars
	float *device_mIn1;
	float *device_mIn2;
	float *device_mOut;

	// alloc GPU memory
	gpuErrchk(cudaMalloc((void**)&device_mIn1, arrayBytes));
	gpuErrchk(cudaMalloc((void**)&device_mIn2, arrayBytes));
	gpuErrchk(cudaMalloc((void**)&device_mOut, arrayBytes));
	// copy memory to device
	gpuErrchk(cudaMemcpy(device_mIn1, host_mIn1, arrayBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(device_mIn2, host_mIn2, arrayBytes, cudaMemcpyHostToDevice));

	// launch kernel
	//kernel << <arraySize, 1 >> >(device_mIn1, device_mIn2, device_mOut);
	// launch kernel with error
	kernelPlusError << <arraySize, 1 >> >(device_mIn1, device_mIn2, device_mOut);
	gpuErrchk(cudaPeekAtLastError()); // debug

	// copy memory from device
	gpuErrchk(cudaMemcpy(host_mOut, device_mOut, arrayBytes, cudaMemcpyDeviceToHost));

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

int StartKernel_floatWithCounter()
{
	Matrix m1("matrixes/bcsstk03.mtx");
	Matrix m2("matrixes/bcsstk03.mtx");

	int arraySize = m1.getNonZeroValuesAmount();
	int arrayBytes = arraySize * sizeof(float);
	int tally = 0; // thread counter

	// init CPU vars // no smart pointers in .cu allowed, watch out
	float *host_mIn1 = new float[arraySize];  CellInfoToFloat(host_mIn1, m1.getMatrix(), m1.getNonZeroValuesAmount());
	float *host_mIn2 = new float[arraySize];  CellInfoToFloat(host_mIn2, m2.getMatrix(), m2.getNonZeroValuesAmount());
	float *host_mOut = new float[arraySize];

	// init GPU vars
	float *device_mIn1;
	float *device_mIn2;
	float *device_mOut;
	int *device_tally;

	// alloc GPU memory
	gpuErrchk(cudaMalloc((void**)&device_mIn1, arrayBytes));
	gpuErrchk(cudaMalloc((void**)&device_mIn2, arrayBytes));
	gpuErrchk(cudaMalloc((void**)&device_mOut, arrayBytes));
	gpuErrchk(cudaMalloc((void **)&device_tally, sizeof(int)));
	// copy memory to device
	gpuErrchk(cudaMemcpy(device_mIn1, host_mIn1, arrayBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(device_mIn2, host_mIn2, arrayBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(device_tally, &tally, sizeof(int), cudaMemcpyHostToDevice));

	// launch kernel
	//kernel << <arraySize, 1 >> >(device_mIn1, device_mIn2, device_mOut);
	// launch kernel with error
	kernelWithCounter << <arraySize, 1 >> >(device_mIn1, device_mIn2, device_mOut, device_tally);
	gpuErrchk(cudaPeekAtLastError()); // debug

	// copy memory from device
	gpuErrchk(cudaMemcpy(host_mOut, device_mOut, arrayBytes, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&tally, device_tally, sizeof(int), cudaMemcpyDeviceToHost));

	// cleaning
	cudaFree(device_mIn1);
	cudaFree(device_mIn2);
	cudaFree(device_mOut);
	cudaFree(device_tally);
	delete[] host_mOut;

	return tally;
}


int StartKernel_floatWithCounter(int gridSize, int blockSize, Matrix &m1, Matrix &m2)
{
	// IMPORTANT - m1 AND m2 NEED TO HAVE EQUAL NON ZERO VALUES AMOUNT

	int arraySize = m1.getNonZeroValuesAmount();
	int arrayBytes = arraySize * sizeof(float);
	int tally = 0; // thread counter

	// init CPU vars // no smart pointers in .cu allowed, watch out
	float *host_mIn1 = new float[arraySize];  CellInfoToFloat(host_mIn1, m1.getMatrix(), m1.getNonZeroValuesAmount());
	float *host_mIn2 = new float[arraySize];  CellInfoToFloat(host_mIn2, m2.getMatrix(), m2.getNonZeroValuesAmount());
	float *host_mOut = new float[arraySize];

	// init GPU vars
	float *device_mIn1;
	float *device_mIn2;
	float *device_mOut;
	int *device_tally;

	// alloc GPU memory
	gpuErrchk(cudaMalloc((void**)&device_mIn1, arrayBytes));
	gpuErrchk(cudaMalloc((void**)&device_mIn2, arrayBytes));
	gpuErrchk(cudaMalloc((void**)&device_mOut, arrayBytes));
	gpuErrchk(cudaMalloc((void **)&device_tally, sizeof(int)));
	// copy memory to device
	gpuErrchk(cudaMemcpy(device_mIn1, host_mIn1, arrayBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(device_mIn2, host_mIn2, arrayBytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(device_tally, &tally, sizeof(int), cudaMemcpyHostToDevice));

	// launch kernel
	//kernel << <arraySize, 1 >> >(device_mIn1, device_mIn2, device_mOut);
	// launch kernel with error
	kernelWithCounter << <gridSize, blockSize >> >(device_mIn1, device_mIn2, device_mOut, device_tally);
	gpuErrchk(cudaPeekAtLastError()); // debug

	// copy memory from device
	gpuErrchk(cudaMemcpy(host_mOut, device_mOut, arrayBytes, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&tally, device_tally, sizeof(int), cudaMemcpyDeviceToHost));

	// cleaning
	cudaFree(device_mIn1);
	cudaFree(device_mIn2);
	cudaFree(device_mOut);
	cudaFree(device_tally);
	delete[] host_mOut;

	return tally;
}