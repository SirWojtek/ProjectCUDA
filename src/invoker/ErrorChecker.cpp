#include "ErrorChecker.hpp"

#include "errorCheckKernel/errorCheckKernel.cuh"
#include "kernel/kernel.cuh"
#include "kernelCommon/gpuErrchk.cuh"

ErrorChecker::ErrorChecker(MatrixData& hostMatrix1, MatrixData& hostMatrix2,
	unsigned arraySize) :
		hostMatrix1_(hostMatrix1),
		hostMatrix2_(hostMatrix2),
		deviceTable1_(NULL),
		deviceTable2_(NULL),
		deviceOutputTable_(NULL),
		boolOutputTable_(NULL),
		arraySize_(arraySize) {}

ErrorChecker::~ErrorChecker()
{
	cudaFree(deviceTable1_);
	cudaFree(deviceTable2_);
	cudaFree(deviceOutputTable_);
	delete[] boolOutputTable_;
}

void ErrorChecker::init()
{
	allocateMemory();
	copyToDevice();
	runAddingKernel();
}

void ErrorChecker::allocateMemory()
{
	gpuErrchk(cudaMalloc((void**)&deviceTable1_, arraySize_ * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&deviceTable2_, arraySize_ * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&deviceOutputTable_, arraySize_ * sizeof(float)));
	boolOutputTable_ = new bool[arraySize_ * sizeof(bool)];
}

void ErrorChecker::copyToDevice()
{
	gpuErrchk(cudaMemcpy(deviceTable1_, hostMatrix1_.getRawTable(),
		arraySize_ * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(deviceTable2_, hostMatrix2_.getRawTable(),
		arraySize_ * sizeof(float), cudaMemcpyHostToDevice));
}

void ErrorChecker::runAddingKernel()
{
	const dim3 gridSize(arraySize_, 1, 1);
	const dim3 blockSize(1, 1, 1);

	// using of kernel invocator wrapper
	runKernel(gridSize, blockSize, deviceTable1_, deviceTable2_, deviceOutputTable_);
	gpuErrchk(cudaPeekAtLastError()); // debugging GPU, handy
}

std::vector<std::pair<unsigned, unsigned>> ErrorChecker::getErrorPosition(MatrixData& hostOutputMatrix)
{
	std::vector<std::pair<unsigned, unsigned>> result;
	runCheckerKernel(hostOutputMatrix.getRawTable());

	for (unsigned i = 0; i < arraySize_; i++)
	{
		if (boolOutputTable_[i])
		{
			result.push_back(hostOutputMatrix.positionVector[i]);
		}
	}

	return result;
}

void ErrorChecker::runCheckerKernel(float* outputTable)
{
	bool* cudaBoolTable;
	float* deviceOutputTable;
	gpuErrchk(cudaMalloc((void**)&cudaBoolTable, arraySize_ * sizeof(bool)));
	gpuErrchk(cudaMalloc((void**)&deviceOutputTable, arraySize_ * sizeof(float)));
	gpuErrchk(cudaMemcpy(deviceOutputTable, outputTable, arraySize_ * sizeof(float), cudaMemcpyHostToDevice));

	const dim3 gridSize(arraySize_, 1, 1);
	const dim3 blockSize(1, 1, 1);

	// using of kernel invocator wrapper
	runErrorCheckKernel(gridSize, blockSize,
		deviceOutputTable, deviceOutputTable_, cudaBoolTable);
	gpuErrchk(cudaPeekAtLastError()); // debugging GPU, handy

	gpuErrchk(cudaMemcpy(boolOutputTable_, cudaBoolTable, arraySize_ * sizeof(bool), cudaMemcpyDeviceToHost));
	cudaFree(cudaBoolTable);
	cudaFree(deviceOutputTable);
}
