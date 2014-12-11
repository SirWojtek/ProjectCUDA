#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// extern calls not supported, needed to rewrite
__device__ int getIdx_1D_1D();
__device__ int getIdx_2D_2D();

__global__ void errorCheckKernel(const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, bool * const d_outputMatrix);

void runErrorCheckKernel(dim3 gridSize, dim3 blockSize, const float  * const d_inputMatrix1,
	const float  * const d_inputMatrix2, bool * const d_outputMatrix);

// for lazy testing
void testErrorCheck();