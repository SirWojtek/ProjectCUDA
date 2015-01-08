#pragma once

#include "MatrixData.hpp"
#include <utility>

class ErrorChecker
{
public:
	ErrorChecker(MatrixData& hostMatrix1, MatrixData& hostMatrix2, unsigned arraySize);
	~ErrorChecker();

	void init();
	std::pair<unsigned, unsigned> getErrorPosition(MatrixData& hostOutputMatrix);

private:
	void allocateMemory();
	void copyToDevice();
	void runAddingKernel();
	void runCheckerKernel(float* outputTable);

	MatrixData& hostMatrix1_;
	MatrixData& hostMatrix2_;

	float* deviceTable1_;
	float* deviceTable2_;
	float* deviceOutputTable_;
	bool* boolOutputTable_;
	unsigned arraySize_;
};