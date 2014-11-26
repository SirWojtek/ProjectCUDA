#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

#include "matrix.hpp"

Matrix::Matrix(std::string filename)
{
	std::ifstream fin(filename);
	int M, N, L;
	while (fin.peek() == '%') fin.ignore(2048, '\n'); // Ignore comments in .mtx file

	fin >> rows_ >> columns_ >> nonZeroValues_;

	matrix_ = new double[rows_*columns_];
	std::fill(matrix_, matrix_ + rows_*columns_, 0.);

	// Read the data
	for (int l = 0; l < nonZeroValues_; l++)
	{
		int row, col;
		double data;
		fin >> row >> col >> data;
		matrix_[(col-1) + (row-1)*rows_] = data;
	}

	fin.close();
}

Matrix::~Matrix()
{
	delete[] matrix_;
}

int Matrix::getRows()
{
	return rows_;
}

int Matrix::getColumns()
{
	return columns_;
}

int Matrix::getNonZeroValuesAmount()
{
	return nonZeroValues_;
}

double Matrix::getV(int row, int col)
{
	if (row == 0 || col == 0){
		std::cout << "Matrix::getV - either row or column argument is zero" << std::endl;
		return -1;
	}
	if ((row*col) > (rows_*columns_))
	{
		std::cout << "Matrix::getV - either row or column argument is greater than rows_ or columns_" << std::endl;
		return -1;
	}
	int arrayPos = (row-1)*rows_ + col - 1;
	return matrix_[arrayPos];
}