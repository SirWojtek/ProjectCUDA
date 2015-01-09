#include "KernelInvoker.hpp"
#include "ErrorChecker.hpp"

#include <exception>
#include <assert.h>
#include <algorithm>
#include <iostream>

#include "device_launch_parameters.h"
#include "../matrix_loader/matrix.hpp"
#include "kernel/kernel.cuh"
#include "kernelCommon/gpuErrchk.cuh"

KernelInvoker::KernelInvoker(unsigned maxThreadNumber) :
	maxThreadNumber_(maxThreadNumber)
{
	redundantData_ = new float[maxThreadNumber_];
}

KernelInvoker::~KernelInvoker()
{
	delete[] redundantData_;
}

Matrix KernelInvoker::compute(const Matrix& m1, const Matrix& m2)
{
	std::cout << "Matrix adding started" << std::endl;

	if (!areMatrixesEqual(m1, m2))
	{
		throw std::runtime_error("Matrix dimensions are not equal");
	}

	init(m1, m2);
	runKernels();

	// This computation is only for check if error was corrected
	// There is no need to compute timing for it or run it on stream
	std::cout << "Computation done, checking for errors" << std::endl;
	checkForErrors();
	
	return getOutputMatrix(m1.getRows(), m1.getColumns());
}

bool KernelInvoker::areMatrixesEqual(const Matrix& m1, const Matrix& m2)
{
	return (m1.getColumns() == m2.getColumns() && m1.getRows() == m2.getRows());
}

void KernelInvoker::init(const Matrix& m1, const Matrix& m2)
{
	readDataFromMatrixes(m1, m2);
	hostOutputMatrix_.resize(arraySize_);
}

void KernelInvoker::readDataFromMatrixes(const Matrix& m1, const Matrix& m2)
{
	readDataFromMatrix(m1, hostInputMatrix1_);
	readDataFromMatrix(m2, hostInputMatrix2_);

	makeSameLength(hostInputMatrix1_, hostInputMatrix2_);
	arraySize_ = hostInputMatrix1_.dataVector.size();
	hostOutputMatrix_.positionVector = hostInputMatrix1_.positionVector;
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

void KernelInvoker::runKernels()
{
	for (unsigned i = 0; i < arraySize_; i += maxThreadNumber_)
	{
		unsigned remaining = arraySize_ - i;
		unsigned threadNumber;
		unsigned redundantThreadNumber;

		if (remaining > maxThreadNumber_)
		{
			threadNumber = maxThreadNumber_;
			redundantThreadNumber = 0;
		}
		else
		{
			threadNumber = remaining;
			redundantThreadNumber =
				maxThreadNumber_ - remaining > remaining ? remaining : maxThreadNumber_ - remaining;
		}

		std::cout << "Remaining jobs: " << remaining << std::endl;
		std::cout << "Thread for computation: " << threadNumber << std::endl;
		std::cout << "Thread for redundant computing: " << redundantThreadNumber << std::endl;

		const dim3 gridSize(threadNumber, 1, 1);
		const dim3 redundantGridSize(redundantThreadNumber, 1, 1);

		// using of kernel invocator wrapper
		runCommandCenter(gridSize, redundantGridSize, arraySize_,
			hostInputMatrix1_.getRawTable(), hostInputMatrix2_.getRawTable(),
			hostOutputMatrix_.getRawTable(), redundantData_);
		gpuErrchk(cudaPeekAtLastError()); // debugging GPU, handy

		correctErrors(hostOutputMatrix_, redundantData_, redundantThreadNumber);
	}
}

void KernelInvoker::correctErrors(MatrixData& matrixWithError, float* redundantMatrix, unsigned redundantSize)
{
	for (unsigned i = 0; i < redundantSize; i++)
	{
		if (matrixWithError.dataVector[i] - redundantMatrix[i] > 0.001)
		{
			const unsigned& positionX = matrixWithError.positionVector[i].first;
			const unsigned& positionY = matrixWithError.positionVector[i].second;

			std::cout << "Corrected error on [ " << positionX << " " << positionY << " ]" << std::endl;
			matrixWithError.dataVector[i] = redundantMatrix[i];
		}
	}
}

void KernelInvoker::checkForErrors()
{
	ErrorChecker checker(hostInputMatrix1_, hostInputMatrix2_, arraySize_);
	checker.init();

	std::pair<unsigned, unsigned> errorPosition = checker.getErrorPosition(hostOutputMatrix_);

	if (errorPosition.first == -1 || errorPosition.second == -1)
	{
		std::cout << "No error detected" << std::endl;
		return;
	}
	
	std::cout << "Error detected at position [ "
		<< errorPosition.first << ", " << errorPosition.second << " ]" << std::endl;
}

Matrix KernelInvoker::getOutputMatrix(unsigned rowNo, unsigned colNo)
{
	CellInfo* info = new CellInfo[arraySize_];
	float* rawTable = hostOutputMatrix_.getRawTable();
	std::vector<float>::iterator it;

	for (unsigned i = 0; i < arraySize_; i++)
	{
		info[i].value = rawTable[i];
		info[i].row = hostOutputMatrix_.positionVector[i].first;
		info[i].column = hostOutputMatrix_.positionVector[i].second;
	}

	return Matrix(info, rowNo, colNo, arraySize_);
}
