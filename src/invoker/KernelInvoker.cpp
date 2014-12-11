
#include "KernelInvoker.hpp"

#include <exception>
#include <fstream>
#include <assert.h>
#include <algorithm>

#include "device_launch_parameters.h"
#include "../matrix_loader/matrix.hpp"
#include "kernel/kernel.cuh"
#include "kernelCommon/gpuErrchk.cuh"

KernelInvoker::KernelInvoker(dim3 blockSize, float redundant) :
	blockSize_(blockSize), redundant_(redundant),
	deviceTable1_(NULL), deviceTable2_(NULL) ,deviceOutputTable_(NULL) {}

KernelInvoker::~KernelInvoker()
{
	cudaFree(deviceTable1_);
	cudaFree(deviceTable2_);
	cudaFree(deviceOutputTable_);
}

void KernelInvoker::compute(const Matrix& m1, const Matrix& m2)
{
	// Make sure arrays have same dimensions
	if (!areMatrixesEqual(m1, m2))
	{
		throw std::runtime_error("Matrix dimensions are not equal");
	}

	init(m1, m2);
	runKernels();
	
	Matrix mOutput = getOutputMatrix(m1.getRows(), m1.getColumns());

	// TODO: write method in Matrix thats print matrixes (operator <<)
	// std::cout << mOutput;
}

bool KernelInvoker::areMatrixesEqual(const Matrix& m1, const Matrix& m2)
{
	return (m1.getColumns() == m2.getColumns() && m1.getRows() == m2.getRows());
}

void KernelInvoker::init(const Matrix& m1, const Matrix& m2)
{
	readDataFromMatrixes(m1, m2);
	hostOutputMatrix_.resize(arraySize_);

	initDevice();
	copyToDevice();
}

void KernelInvoker::readDataFromMatrixes(const Matrix& m1, const Matrix& m2)
{
	readDataFromMatrix(m1, hostInputMatrix1_);
	readDataFromMatrix(m2, hostInputMatrix2_);

	makeSameLength(hostInputMatrix1_, hostInputMatrix2_);
	arraySize_ = hostInputMatrix1_.dataVector.size();
}

void KernelInvoker::readDataFromMatrix(const Matrix& m1, MatrixData& output)
{
	CellInfo* data = m1.getMatrix();

	for (unsigned i = 0; i < m1.getNonZeroValuesAmount(); i++)
	{
		output.addElement(data[i].value, data[i].row, data[i].column);
	}
}

void KernelInvoker::makeSameLength(MatrixData& m1, MatrixData& m2)
{
	addZeroValuesOnNotExistingPosition(m1, m2);
	addZeroValuesOnNotExistingPosition(m2, m1);
}

void KernelInvoker::addZeroValuesOnNotExistingPosition(MatrixData& m1, MatrixData& m2)
{
	for (unsigned i = 0; i < m1.positionVector.size(); i++)
	{
		float& currentValue = m1.dataVector[i];
		std::pair<unsigned, unsigned>& currentPos = m1.positionVector[i];

		if (std::find(m2.positionVector.begin(), m2.positionVector.end(), currentPos) ==
			m2.positionVector.end())
		{
			m2.insert(i, currentValue, currentPos.first, currentPos.second);
		}
	}
}

void KernelInvoker::initDevice()
{
	arrayBytes_ = arraySize_ * sizeof(float);

	gpuErrchk(cudaMalloc((void**)&deviceTable1_, arrayBytes_));
	gpuErrchk(cudaMalloc((void**)&deviceTable2_, arrayBytes_));
	gpuErrchk(cudaMalloc((void**)&deviceOutputTable_, arrayBytes_));
}

void KernelInvoker::copyToDevice()
{
	gpuErrchk(cudaMemcpy(deviceTable1_, hostInputMatrix1_.getRawTable(), arrayBytes_, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(deviceTable2_, hostInputMatrix2_.getRawTable(), arrayBytes_, cudaMemcpyHostToDevice));
}

void KernelInvoker::runKernels()
{
	const dim3 gridSize(arraySize_, 1, 1);
	const dim3 blockSize(1, 1, 1);

	// using of kernel invocator wrapper
	runKernelPlusError(gridSize, blockSize, deviceTable1_, deviceTable2_, deviceOutputTable_);
	gpuErrchk(cudaPeekAtLastError()); // debugging GPU, handy
	copyResultToHost();
}

void KernelInvoker::copyResultToHost()
{
	gpuErrchk(cudaMemcpy(hostOutputMatrix_.getRawTable(), deviceOutputTable_, arrayBytes_, cudaMemcpyDeviceToHost));
}

Matrix KernelInvoker::getOutputMatrix(unsigned rowNo, unsigned colNo)
{
	CellInfo* info = new CellInfo[arraySize_];
	float* rawTable = hostOutputMatrix_.getRawTable();
	std::vector<float>::iterator it;

	for (unsigned i = 0; i < arraySize_; i++)
	{
		info[i].value = rawTable[i];
		info[i].row = hostOutputMatrix_.positionVector[0].first;
		info[i].column = hostOutputMatrix_.positionVector[0].second;
	}

	return Matrix(info, rowNo, colNo, arraySize_);
}

bool KernelInvoker::isResultCorrect_Add(const Matrix& m1, const Matrix& m2, const Matrix& mResult)
{
	// TODO: write kernel, that checks if two vectors are identical
	return false;
}