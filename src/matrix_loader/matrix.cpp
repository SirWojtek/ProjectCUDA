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

Matrix::Matrix(const Matrix &object)
{
	this->rows_ = object.getRows();
	this->columns_ = object.getColumns();
	this->nonZeroValues_ = object.getNonZeroValuesAmount();
	this->matrix_ = new double[rows_ * columns_];

	for (int i = 0; i < rows_*columns_; i++)
	{
		this->matrix_[i] = object.matrix_[i];
	}

}


Matrix& Matrix::operator=(Matrix rhs)
{
	this->swap(*this, rhs);
	return *this;
}

Matrix& Matrix::operator+=(const Matrix &rhs)
{
	for (int i = 0; i < columns_*rows_; i++)
		matrix_[i] += rhs.matrix_[i];
	return *this;
}


void Matrix::swap(Matrix &matrix1, Matrix &matrix2)
{
	using std::swap;

	swap(matrix1.columns_, matrix2.columns_);
	swap(matrix1.rows_, matrix2.rows_);
	swap(matrix1.nonZeroValues_, matrix2.nonZeroValues_);
	swap(matrix1.matrix_, matrix2.matrix_);
}

Matrix::~Matrix()
{
	delete[] matrix_;
}

int Matrix::getRows() const
{
	return rows_;
}

int Matrix::getColumns() const
{
	return columns_;
}

int Matrix::getNonZeroValuesAmount() const
{
	return nonZeroValues_;
}

double Matrix::getV(int row, int col) const
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

double * Matrix::getMatrix() const // PB
{
	return this->matrix_;
}