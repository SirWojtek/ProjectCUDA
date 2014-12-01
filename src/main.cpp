#include <iostream>
#include <string>
#include <exception>

#include "device_launch_parameters.h"
#include "invoker\KernelInvoker.hpp"

void run()
{
	Matrix h_in1("matrixes/bcsstk03.mtx");
	Matrix h_in2("matrixes/bcsstk03.mtx");
	Matrix error("matrixes/bcsstk03.mtx");

	dim3 gridSize(h_in1.getColumns(), h_in2.getRows(),1);
	dim3 blockSize(1, 1, 1);

	KernelInvoker invoker(gridSize, 0.0);

	invoker.compute(h_in1, h_in2, error);
}

int main()
{
	try
	{
		run();
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught std exception: " << e.what();
	}
	catch (const std::string& e)
	{
		std::cerr << "Caught string exception: " << e;
	}
	catch (...)
	{
		std::cerr << "Caught unknown exception";
	}

	std::cout << std::endl;
	system("pause");
}
