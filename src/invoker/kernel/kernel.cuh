#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// get threads global id
__device__ int getGlobalIdx_2D_2D();

// sum operation on two matrices
__device__ float sumMatrix(const float * const d_inputMatrix1, const float * const d_inputMatrix2, int index);

// injecting error into matrix
__device__ void injectError(float * const d_inputMatrix, const float * const d_errorMap, int index);

// kernel performing operations on two matrices ver without error injection
__global__ void matrixOperation(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix);

// kernel performing operations on two matrices ver with error injection
__global__ void matrixOperation(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix, float * const d_errorMap);

// fill error map with values (depracated)
extern void fillErrorMap(float * const errorMap, const int numRows, const int numCols);

// wrappers for C++ class calling kernel
void runKernel(dim3 gridSize, dim3 blockSize, float* in1, float* in2, float* out);
void runKernelWithError(dim3 gridSize, dim3 blockSize, float* in1, float* in2,
	float* out, float* error);
