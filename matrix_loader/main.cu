#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matrix.hpp"

int main()
{ 
	Matrix* x = new Matrix("bcsstk03.mtx");
	std::cout << "Examples for bad behaviour:" << std::endl;
	std::cout << x->getV(0,0) << std::endl;
	std::cout << x->getV(122,122) << std::endl;
	std::cout << "____________________________" << std::endl;
	std::cout << "Examples for good behaviour:" << std::endl;
	std::cout << x->getV(1,1) << std::endl; // first element, which we would normally get by Matrix[0][0]
	std::cout << x->getV(4,1) << std::endl;
	std::cout << x->getV(112,112) << std::endl;
	std::cout << "Check bcsstk03.mtx file for correctness of these values" << std::endl;

	delete x;
}
