#include "deviceProperties.cuh"

#include <iostream>
#include "cuda_runtime.h"

deviceProperties::deviceProperties() : maxNumberOfThreads_(0), deviceCount_(0)
{
	checkDevices();
	checkMaxNumberOfThreads();
}

void deviceProperties::printDeviceProperties() const
{
	std::cout << "Number of devices: " << deviceCount_ << std::endl;
	std::cout << "Max number of threads: " << maxNumberOfThreads_ << std::endl;
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

