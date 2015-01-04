#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class Matrix;

class deviceProperties
{
public:
	CUDA_CALLABLE_MEMBER deviceProperties();
	inline int getNumberOfDevices() const { return deviceCount_; }
	inline int getMaxNumberOfThreads() const { return maxNumberOfThreads_; }
	CUDA_CALLABLE_MEMBER void printDeviceProperties() const;
	CUDA_CALLABLE_MEMBER int checkNumberOfThreadsExecuted();	
	CUDA_CALLABLE_MEMBER int checkNumberOfThreadsExecuted(int gridSize, int blockSize, Matrix &m1, Matrix &m2);
private:
	int maxNumberOfThreads_;
	int deviceCount_;

	void checkDevices();
	void checkMaxNumberOfThreads();
};

void devicePropertiesExample();