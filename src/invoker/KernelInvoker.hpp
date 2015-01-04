#pragma once

#include "../matrix_loader/matrix.hpp"
#include "device_launch_parameters.h"
#include <vector>
#include <utility>

struct MatrixData
{
	std::vector<float> dataVector;
	std::vector<std::pair<unsigned, unsigned> > positionVector;

	float* getRawTable()
	{
		return &dataVector[0];
	}

	void addElement(float data, unsigned row, unsigned column)
	{
		dataVector.push_back(data);
		positionVector.push_back(std::make_pair(row, column));
	}

	void insert(unsigned position, float data, unsigned row, unsigned column)
	{
		dataVector.insert(dataVector.begin() + position, data);
		positionVector.insert(positionVector.begin() + position, std::make_pair(row, column));
	}

	void resize(unsigned newSize)
	{
		dataVector.resize(newSize);
		positionVector.resize(newSize);
	}
};

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
	void initDevice();
	void copyToDevice();
	void runKernels();
	void copyResultToHost();
	void checkForErrors();
	void printErrorPosition(unsigned errorPos);
	Matrix getOutputMatrix(unsigned rowNo, unsigned colNo);
	bool isResultCorrect_Add(const Matrix& m1, const Matrix& m2, const Matrix& mResult);
	
	MatrixData hostInputMatrix1_;
	MatrixData hostInputMatrix2_;
	MatrixData hostOutputMatrix_;

	unsigned arraySize_, arrayBytes_;
	float* deviceTable1_;
	float* deviceTable2_;
	float* deviceOutputTable_;

	unsigned maxThreadNumber_;
};
