#pragma once

#include "../matrix_loader/matrix.hpp"
#include "device_launch_parameters.h"
#include "MatrixData.hpp"

class KernelInvoker
{
public:
	KernelInvoker(unsigned maxThreadNumber);
	~KernelInvoker();
	Matrix compute(const Matrix& m1, const Matrix& m2);

private:
	bool areMatrixesEqual(const Matrix& m1, const Matrix& m2);
	void init(const Matrix& m1, const Matrix& m2);
	void readDataFromMatrixes(const Matrix& m1, const Matrix& m2);
	void readDataFromMatrix(const Matrix& m1, MatrixData& output);
	void makeSameLength(MatrixData& m1, MatrixData& m2);
	void addZeroValuesOnNotExistingPosition(MatrixData& m1, MatrixData& m2);
	void runKernels();
	void correctErrors(unsigned redundantSize);
	void checkForErrors();
	Matrix getOutputMatrix(unsigned rowNo, unsigned colNo);

	MatrixData hostInputMatrix1_;
	MatrixData hostInputMatrix2_;
	MatrixData hostOutputMatrix_;
	float * redundantData_;
	unsigned arraySize_;
	unsigned maxThreadNumber_;
};
