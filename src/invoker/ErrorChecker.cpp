#include "ErrorChecker.hpp"

#include "errorCheckKernel/errorCheckKernel.cuh"
#include "kernel/kernel.cuh"
#include "kernelCommon/gpuErrchk.cuh"

ErrorChecker::ErrorChecker(float* deviceTable1, float* deviceTable2,
	unsigned arraySize, float delta) :
		deviceTable1_(deviceTable1),
		deviceTable2_(deviceTable2),
		deviceOutputTable_(NULL),
		boolOutputTable_(NULL),
		arraySize_(arraySize),
		delta_(delta) {}

ErrorChecker::~ErrorChecker()
{
	cudaFree(deviceOutputTable_);
	delete[] boolOutputTable_;
}

void ErrorChecker::init()
{
	allocateMemory();
	runAddingKernel();
}

void ErrorChecker::allocateMemory()
{
	gpuErrchk(cudaMalloc((void**)&deviceOutputTable_, arraySize_ * sizeof(float)));
	boolOutputTable_ = new bool[arraySize_ * sizeof(bool)];
}

void ErrorChecker::runAddingKernel()
{
	const dim3 gridSize(arraySize_, 1, 1);
	const dim3 blockSize(1, 1, 1);

	// using of kernel invocator wrapper
	runKernel(gridSize, blockSize, deviceTable1_, deviceTable2_, deviceOutputTable_);
	gpuErrchk(cudaPeekAtLastError()); // debugging GPU, handy
}

int ErrorChecker::getErrorPosition(float* deviceOutputTable)
{
	runCheckerKernel(deviceOutputTable);

	for (unsigned i = 0; i < arraySize_; i++)
	{
		if (boolOutputTable_[i])
		{
			return i;
		}
	}

	return -1;
}

void ErrorChecker::runCheckerKernel(float* deviceOutputTable)
{
	bool* cudaBoolTable;
	gpuErrchk(cudaMalloc((void**)&cudaBoolTable, arraySize_ * sizeof(bool)));

	const dim3 gridSize(arraySize_, 1, 1);
	const dim3 blockSize(1, 1, 1);

	// using of kernel invocator wrapper
	runErrorCheckKernel(gridSize, blockSize,
		deviceOutputTable, deviceOutputTable_, cudaBoolTable);
	gpuErrchk(cudaPeekAtLastError()); // debugging GPU, handy

	gpuErrchk(cudaMemcpy(boolOutputTable_, cudaBoolTable, arraySize_ * sizeof(bool), cudaMemcpyDeviceToHost));
	cudaFree(cudaBoolTable);
}
