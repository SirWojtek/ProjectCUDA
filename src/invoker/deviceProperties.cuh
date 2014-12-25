#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class deviceProperties
{
public:
	CUDA_CALLABLE_MEMBER deviceProperties();
	inline int getNumberOfDevices() const { return deviceCount_; }
	inline int getMaxNumberOfThreads() const { return maxNumberOfThreads_; }
	void printDeviceProperties() const;
private:
	int maxNumberOfThreads_;
	int deviceCount_;

	void checkDevices();
	void checkMaxNumberOfThreads();
};