#include <iostream>
#include <string>
#include <exception>
#include <fstream>

#include "device_launch_parameters.h"
#include "invoker\KernelInvoker.hpp"

#include "invoker\deviceProperties.cuh"

void MatrixTest()
{
	Matrix::matrixIntegrationTest();
	system("pause");
	exit(0);
}

void run()
{
	std::ofstream file("result.log");

	deviceProperties properties;
	properties.printDeviceProperties();

	Matrix h_in1("matrixes/bcsstk03.mtx");
	Matrix h_in2("matrixes/bcsstk03.mtx");
	h_in2.randomize();

	file << h_in1 << std::endl << std::endl << h_in2 << std::endl << std::endl;

	// KernelInvoker invoker(properties.getMaxNumberOfThreads());
	KernelInvoker invoker(100);


	Matrix result = invoker.compute(h_in1, h_in2);
	file << result << std::endl;
}

int main()
{
	// devicePropertiesExample();
	// MatrixTest();
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
