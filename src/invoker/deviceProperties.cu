#include "deviceProperties.cuh"

#include <iostream>
#include "cuda_runtime.h"
#include "kernel\kernel.cuh"
#include "..\matrix_loader\matrix.hpp"

deviceProperties::deviceProperties() : maxNumberOfThreads_(0), deviceCount_(0)
{
	checkDevices();
	checkMaxNumberOfThreads();
}

void deviceProperties::printDeviceProperties() const
{
	std::cout << "Number of devices: " << deviceCount_ << std::endl;
	std::cout << "Max number of threads executed at the same time: " << maxNumberOfThreads_ << std::endl;
}

int deviceProperties::checkNumberOfThreadsExecuted()
{
	return StartKernel_floatWithCounter();
}

int deviceProperties::checkNumberOfThreadsExecuted(int gridSize, int blockSize, Matrix &m1, Matrix &m2)
{
	if (m1.getNonZeroValuesAmount() == m2.getNonZeroValuesAmount())
		return StartKernel_floatWithCounter(gridSize, blockSize, m1, m2);
	else
	{
		std::cout << "for testing number of threads executed simultaneously, both m1 and m2 need to have equal nonzero values amount" << std::endl;
		return 0;
	}
}

void deviceProperties::checkDevices()
{
	int deviceCount;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	if (cudaResultCode != cudaSuccess)
		deviceCount = 0;

	deviceCount_ = deviceCount;
}

void deviceProperties::checkMaxNumberOfThreads()
{
	int device;
	int gpuDeviceCount = 0;
	struct cudaDeviceProp properties;
	
	
	/* machines with no GPUs can still report one emulation device */
	for (device = 0; device < deviceCount_; ++device) {
		cudaGetDeviceProperties(&properties, device);
		if (properties.major != 9999) /* 9999 means emulation only */
			this->maxNumberOfThreads_ += properties.multiProcessorCount*properties.maxThreadsPerMultiProcessor;
	}
}

void devicePropertiesExample()
{
	deviceProperties properties; 
	properties.printDeviceProperties(); 
	Matrix m1("matrixes/olm1000.mtx"); 
	std::cout << "Number of threads executed: " << properties.checkNumberOfThreadsExecuted(3996, 1, m1, m1) << std::endl;

}