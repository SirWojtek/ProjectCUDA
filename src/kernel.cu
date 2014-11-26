
// musialem ustawic compute_11 z 20 w project properties-> cuda c/c++ -> device ->code generation



#include "kernel.cuh"

#include <iostream>
#include <stdio.h>
#include "gpuErrchk.cuh"

__device__ int getGlobalIdx_2D_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

template<class matrix>
__device__ matrix sumMatrix(const matrix * const d_inputMatrix1, const matrix * const d_inputMatrix2, int index)
{
	return d_inputMatrix1[index] + d_inputMatrix2[index];
}

template<class matrix, class error>
__device__ void injectError(matrix * const d_inputMatrix, const error * const d_errorMap, int index)
{
	d_inputMatrix[index] += d_errorMap[index];
}

template<class matrix, class error>
__global__ void matrixOperation(const matrix  * const d_inputMatrix1, const matrix  * const d_inputMatrix2, matrix * const d_outputMatrix, int numRows, const error * const d_errorMap)
{
	int index = getGlobalIdx_2D_2D();
	d_outputMatrix[index] = sumMatrix(d_inputMatrix1, d_inputMatrix2, index);
	injectError(d_outputMatrix, d_errorMap, index);
}

template <class error>
void fillErrorMap(error * const errorMap, const int numRows)
{
	for (int i = 0; i < numRows; i++)
	{
		errorMap[i] = 1;
	}
}

void startKernel()
{
	const int ARRAY_SIZE = 10;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i) + 1;

	}
	float h_out[ARRAY_SIZE];

	float h_error[ARRAY_SIZE];
	fillErrorMap(h_error, ARRAY_SIZE);

	// declare GPU memory pointers
	float * d_in1;
	float * d_in2;
	float * d_out;
	float * d_error;

	// allocate GPU memory
	gpuErrchk(cudaMalloc((void**)&d_in1, ARRAY_BYTES));
	gpuErrchk(cudaMalloc((void**)&d_in2, ARRAY_BYTES));
	gpuErrchk(cudaMalloc((void**)&d_out, ARRAY_BYTES));
	gpuErrchk(cudaMalloc((void**)&d_error, ARRAY_BYTES));

	std::cout << "Matrix = " << std::endl;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		std::cout << h_in[i] << std::endl;

	}


	// transfer the array to the GPU
	gpuErrchk(cudaMemcpy(d_in1, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_in2, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_error, h_error, ARRAY_BYTES, cudaMemcpyHostToDevice));

	// launch the kernel
	matrixOperation <<< 1, ARRAY_SIZE >>>(d_in1, d_in2, d_out, ARRAY_SIZE, d_error);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// copy back the result array to the CPU
	gpuErrchk(cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));

	// print out the resulting array
	std::cout << "Matrix + Matrix + Error =" << std::endl;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		std::cout << h_out[i] << std::endl;
	}


	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);
}

