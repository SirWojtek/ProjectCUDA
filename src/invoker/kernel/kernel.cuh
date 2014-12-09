#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct CellInfo;

// get broken thread's global id
__device__ int getErrorIdx_1D_1D();
// get threads global id
__device__ int getGlobalIdx_1D_1D();
__device__ int getGlobalIdx_2D_2D();

// sum operation on two matrices
__device__ CellInfo sumMatrix(const CellInfo * const d_inputMatrix1, const CellInfo * const d_inputMatrix2, int index);
__device__ float sumMatrix(const float * const d_inputMatrix1, const float * const d_inputMatrix2, int index);

// is there need for error?
__device__ bool isError(int index);

// injecting error into matrix
__device__ void injectError(CellInfo &inputCell);
__device__ void injectError(float * inputCell);


// kernel performing operations on two matrices ver without error injection
__global__ void kernel(const CellInfo  * const d_inputMatrix1,
	const CellInfo  * const d_inputMatrix2, CellInfo * const d_outputMatrix);
__global__ void kernel(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix);


// kernel performing operations on two matrices ver with error injection
__global__ void kernelPlusError(const CellInfo  * const d_inputMatrix1,
	const CellInfo  * const d_inputMatrix2, CellInfo * const d_outputMatrix);
__global__ void kernelPlusError(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, float * const d_outputMatrix);

// wrappers for C++ class calling kernel
//void runKernel(dim3 gridSize, dim3 blockSize, float* in1, float* in2, float* out);
//void runKernelWithError(dim3 gridSize, dim3 blockSize, float* in1, float* in2,
//	float* out, float* error);

void CellInfoToFloat(float * output, CellInfo * input, int arraySize);

void testStartKernel_CellInfo();
void testStartKernel_float();