#pragma once

class ErrorChecker
{
public:
	ErrorChecker(float* deviceTable1, float* deviceTable2,
		unsigned arraySize, float delta);
	~ErrorChecker();

	void init();
	int getErrorPosition(float* deviceOutputTable);

private:
	void allocateMemory();
	void runAddingKernel();
	void runCheckerKernel(float* deviceOutputTable);

	float* deviceTable1_;
	float* deviceTable2_;
	float* deviceOutputTable_;
	bool* boolOutputTable_;
	unsigned arraySize_;
	float delta_;
};