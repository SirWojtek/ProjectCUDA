#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct CellInfo;

// get threads global id
__device__ int getGlobalIdx_1D_1D();
__device__ int getGlobalIdx_2D_2D();

// sum operation on two matrices
__device__ CellInfo sumMatrix(const CellInfo * const d_inputMatrix1, const CellInfo * const d_inputMatrix2, int index);

// is there need for error?
__device__ bool isError(int index);

// injecting error into matrix
__device__ void injectError(CellInfo &inputCell);

// kernel performing operations on two matrices ver without error injection
__global__ void kernel(const CellInfo  * const d_inputMatrix1,
	const CellInfo  * const d_inputMatrix2, CellInfo * const d_outputMatrix);

// kernel performing operations on two matrices ver with error injection
__global__ void kernelPlusError(const CellInfo  * const d_inputMatrix1,
	const CellInfo  * const d_inputMatrix2, CellInfo * const d_outputMatrix);

// wrappers for C++ class calling kernel
//void runKernel(dim3 gridSize, dim3 blockSize, float* in1, float* in2, float* out);
//void runKernelWithError(dim3 gridSize, dim3 blockSize, float* in1, float* in2,
//	float* out, float* error);


void testStartKernel();