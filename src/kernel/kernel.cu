#include "kernel.cuh"

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include "gpuErrchk.cuh"

#include "..\matrix_loader\matrix.hpp"

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
__global__ void matrixOperation(const matrix  * const d_inputMatrix1, const matrix  * const d_inputMatrix2, matrix * const d_outputMatrix, const error * const d_errorMap)
{
	int index = getGlobalIdx_2D_2D();
	d_outputMatrix[index] = sumMatrix(d_inputMatrix1, d_inputMatrix2, index);
	injectError(d_outputMatrix, d_errorMap, index);
}

template <class error>
void fillErrorMap(error * const errorMap, const int numRows, const int numCols)
{
	for (int i = 0; i < numRows*numCols; i++)
	{
		errorMap[i] = 10;
	}
}


void startKernel()
{
	// C++11 not supported by CUDA (can't use smart pointers)
	Matrix * h_in1 = new Matrix("matrixes/bcsstk03.mtx");
	Matrix * h_in2 = new Matrix("matrixes/bcsstk03.mtx");

	// Make sure arrays have same dimensions
	assert(h_in1->getColumns() == h_in2->getColumns());
	assert(h_in1->getRows() == h_in2->getRows());

	const int ARRAY_SIZE = h_in1->getColumns()*h_in1->getRows();
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);


	// CUDA SM 1.1 doesn't support double, need to convert to floats
	float * h_in1_float = new float[ARRAY_SIZE];
	for (int i = 0; i < 112 * 112; i++)
	{
		h_in1_float[i] = static_cast<float>(h_in1->getMatrix()[i]);
	}

	float * h_in2_float = new float[ARRAY_SIZE];
	for (int i = 0; i < 112 * 112; i++)
	{
		h_in2_float[i] = static_cast<float>(h_in1->getMatrix()[i]);
	}



	float * h_out = new float[ARRAY_SIZE];

	float * h_error = new float[ARRAY_SIZE];
	fillErrorMap(h_error, h_in1->getRows(), h_in1->getColumns());


	


	for (int i = 0; i < 10; i++)
	{	std::cout << "h1  -> " << h_in1_float[i] << "; ";
		std::cout << "h2  -> " << h_in2_float[i] << "; ";
		std::cout << "err -> " << h_error[i] << ";";

		std::cout << "suma ->" << h_in1_float[i] + h_in1_float[i] + h_error[i] << std::endl;
	}

	float * d_in1;
	float * d_in2;
	float * d_out;
	float * d_error;
	

	gpuErrchk(cudaMalloc((void**)&d_in1, ARRAY_BYTES));
	gpuErrchk(cudaMalloc((void**)&d_in2, ARRAY_BYTES));
	gpuErrchk(cudaMalloc((void**)&d_out, ARRAY_BYTES));
	gpuErrchk(cudaMalloc((void**)&d_error, ARRAY_BYTES));


	//gpuErrchk(cudaMemcpy(d_in1, h_in1->getMatrix(), ARRAY_BYTES, cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(d_in2, h_in2->getMatrix(), ARRAY_BYTES, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_in1, h_in1_float, ARRAY_BYTES, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_in2, h_in2_float, ARRAY_BYTES, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_error, h_error, ARRAY_BYTES, cudaMemcpyHostToDevice));

	const dim3 gridSize(112, 112, 1);  
	const dim3 blockSize(1, 1, 1);  
	matrixOperation <<< gridSize, blockSize >>>(d_in1, d_in2, d_out, d_error);
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
	
	std::cout << "out:" << std::endl;
	for (int i = 0; i < 112; i++)
		std::cout << "h_out -> " << h_out[i] << std::endl;

	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);
	cudaFree(d_error);


}

