#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// get threads global id
__device__ int getGlobalIdx_2D_2D();

// sum operation on two matrices
template<class matrix>
__device__ matrix sumMatrix(const matrix * const d_inputMatrix1, const matrix * const d_inputMatrix2, int index);

// injecting error into matrix
template<class matrix, class error>
__device__ void injectError(matrix * const d_inputMatrix, const error * const d_errorMap, int index);

// kernel performing operations on two matrices
template<class matrix, class error>
__global__ void matrixOperation(const matrix  * const d_inputMatrix1, const matrix  * const d_inputMatrix2, matrix * const d_outputMatrix, int numRows, const error * const d_errorMap);

// fill error map with values
template <class error>
void fillErrorMap(error * const errorMap);

// example kernel execution
void startKernel();

#endif