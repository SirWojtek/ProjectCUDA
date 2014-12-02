
#include "KernelInvoker.hpp"

#include <exception>
#include <fstream>

#include "device_launch_parameters.h"
#include "../matrix_loader/matrix.hpp"
#include "kernel/kernel.cuh"
#include "kernel/gpuErrchk.cuh"

KernelInvoker::KernelInvoker(dim3 blockSize, float redundant) :
	blockSize_(blockSize), redundant_(redundant),
	hostTable1_(NULL), hostTable2_(NULL) ,hostOutputTable_(NULL), hostErrorTable_(NULL),
	deviceTable1_(NULL), deviceTable2_(NULL) ,deviceOutputTable_(NULL), deviceErrorTable_(NULL) {}

KernelInvoker::~KernelInvoker()
{
	delete[] hostTable1_;
	delete[] hostTable2_;
	delete[] hostOutputTable_;
	delete[] hostErrorTable_;

	cudaFree(deviceTable1_);
	cudaFree(deviceTable2_);
	cudaFree(deviceOutputTable_);
	cudaFree(deviceErrorTable_);
}

void KernelInvoker::compute(const Matrix& m1, const Matrix& m2, const Matrix& errorMatrix)
{
	// Make sure arrays have same dimensions
	if (!areMatrixesEqual(m1, m2, errorMatrix))
	{
		throw std::runtime_error("Matrix dimensions are not equal");
	}

	initHost(m1);
	readToFloatTable(m1, m2, errorMatrix);

	initDevice();
	copyToDevice();

	//const dim3 gridSize = getGridSize();
	const dim3 gridSize(m1.getColumns(), m1.getRows(), 1);
	const dim3 blockSize(1, 1, 1);
	
	runKernel(gridSize, (1, 1 ,1), deviceTable1_, deviceTable2_, deviceOutputTable_);
	gpuErrchk(cudaPeekAtLastError()); // debugging GPU, handy
	copyResultToHost();

	// TODO: w Matrix metoda pozwalajaca wczytac z tablicy dwuwymiarowej
	// zwracanie macierzy jako wyniku
	
	// TODO zrobione. Nie uda³o mi siê tylko obs³ugiwaæ wyj¹tków dla index out of range
	// wiêc póki co podawajcie POPRAWNE wymiary tablic ;)
	Matrix mOutput(this->hostOutputTable_, m1.getColumns(), m1.getRows()); // <- for 1D array

	printResult();
}

bool KernelInvoker::areMatrixesEqual(const Matrix& m1, const Matrix& m2, const Matrix& errorMatrix)
{
	return (m1.getColumns() == m2.getColumns() && m1.getColumns() == errorMatrix.getColumns()) ||
		(m1.getRows() == m2.getRows() && m2.getRows() == errorMatrix.getRows());
}


void KernelInvoker::initHost(const Matrix& m)
{
	arraySize_ = m.getColumns() * m.getRows();

	hostTable1_ = new float[arraySize_];
	hostTable2_ = new float[arraySize_];
	hostOutputTable_ = new float[arraySize_];

	for (int i = 0; i < arraySize_; i++) // filling with 0 for clarity
		hostOutputTable_[i] = 0;

	hostErrorTable_ = new float[arraySize_];
}

void KernelInvoker::readToFloatTable(const Matrix& m1, const Matrix& m2, const Matrix& errorMatrix)
{
	for (int i = 0; i < arraySize_; i++)
	{
		hostTable1_[i] = static_cast<float>(m1.getMatrix()[i]);
		hostTable2_[i] = static_cast<float>(m2.getMatrix()[i]);
		hostErrorTable_[i] = static_cast<float>(errorMatrix.getMatrix()[i]);
	}
}

void KernelInvoker::initDevice()
{
	//arrayBytes_ = getArraySize();
	arrayBytes_ = arraySize_ * sizeof(float);

	gpuErrchk(cudaMalloc((void**)&deviceTable1_, arrayBytes_));
	gpuErrchk(cudaMalloc((void**)&deviceTable2_, arrayBytes_));
	gpuErrchk(cudaMalloc((void**)&deviceOutputTable_, arrayBytes_));
	gpuErrchk(cudaMalloc((void**)&deviceErrorTable_, arrayBytes_));

	// not needed?
	//gpuErrchk(cudaMemset((void**)&deviceTable1_, 0, arrayBytes_));
	//gpuErrchk(cudaMemset((void**)&deviceTable2_, 0, arrayBytes_));
	//gpuErrchk(cudaMemset((void**)&deviceOutputTable_, 0, arrayBytes_));
	//gpuErrchk(cudaMemset((void**)&deviceErrorTable_, 0, arrayBytes_));
}

unsigned KernelInvoker::getArraySize()
{
	return arraySize_ % (blockSize_.x * blockSize_.y) ?
		(arraySize_ + blockSize_.x * blockSize_.y) * sizeof(float) :
		arraySize_ * sizeof(float);
}

void KernelInvoker::copyToDevice()
{
	gpuErrchk(cudaMemcpy(deviceTable1_, hostTable1_, arrayBytes_, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(deviceTable2_, hostTable2_, arrayBytes_, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(deviceErrorTable_, hostErrorTable_, arrayBytes_, cudaMemcpyHostToDevice));
}

dim3 KernelInvoker::getGridSize()
{
	return dim3(arraySize_ / (blockSize_.x * blockSize_.y));
}

void KernelInvoker::copyResultToHost()
{
	gpuErrchk(cudaMemcpy(hostOutputTable_, deviceOutputTable_, arrayBytes_, cudaMemcpyDeviceToHost));
}

void KernelInvoker::printResult()
{
	std::ofstream f("result.log");
	
	f << "Matrix 1:" << std::endl;

	for (int i = 0; i < arraySize_; i++)
	{
		f << hostTable1_[i] << " ";
	}

	f << std::endl << "Matrix 2:" << std::endl;

	for (int i = 0; i < arraySize_; i++)
	{
		f << hostTable2_[i] << " ";
	}

	f << std::endl << "Output matrix:" << std::endl;

	for (int i = 0; i < arraySize_; i++)
	{
		f << hostOutputTable_[i] << " ";
	}
}
