#include <iostream>
#include <memory>

#include "matrix_loader\matrix.hpp"
#include "kernel\kernel.cuh"

using namespace std;

int main()
{
	shared_ptr<Matrix> x = make_shared<Matrix>("matrixes/bcsstk03.mtx");
	cout << "Examples for bad behaviour:" << endl;
	cout << x->getV(0, 0) << endl;
	cout << x->getV(122, 122) << endl;
	cout << "____________________________" << endl;
	cout << "Examples for good behaviour:" << endl;
	cout << x->getV(1, 1) << endl; // first element, which we would normally get by Matrix[0][0]
	cout << x->getV(4, 1) << endl;
	cout << x->getV(112, 112) << endl;
	cout << "Check bcsstk03.mtx file for correctness of these values" << endl;

	cout << endl;
	startKernel();

	cout << endl;
	system("pause");
}
