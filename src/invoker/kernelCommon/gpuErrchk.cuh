#pragma once

#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		std::ostringstream s;
		s << "GPUassert: " << cudaGetErrorString(code) << " " << file << " "
			<< line << " " << std::endl;
		throw s.str();
	}
}
