#pragma once

#include "../matrix_loader/matrix.hpp"
#include "device_launch_parameters.h"


class KernelInvoker
{
public:
	KernelInvoker(dim3 blockSize, float redundantPercent);
	~KernelInvoker();
	void compute(const Matrix& m1, const Matrix& m2, const Matrix& errorMatrix);

private:
	bool areMatrixesEqual(const Matrix& m1, const Matrix& m2, const Matrix& errorMatrix);
	void initHost(const Matrix& m);
	void readToFloatTable(const Matrix& m1, const Matrix& m2, const Matrix& errorMatrix);
	void initDevice();
	unsigned getArraySize();
	void copyToDevice();
	dim3 getGridSize();
	void copyResultToHost();
	void printResult();

	unsigned arraySize_, arrayBytes_;
	float* hostTable1_;
	float* hostTable2_;
	float* hostOutputTable_;
	float* hostErrorTable_;

	float* deviceTable1_;
	float* deviceTable2_;
	float* deviceOutputTable_;
	float* deviceErrorTable_;

	dim3 blockSize_;
	float redundant_;
};
