#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <windows.h>

#include "matrix.hpp"

Matrix::Matrix(std::string filename)
{
	std::ifstream fin(filename);
	if (!fin.good())
	{
		return;
	}
	while (fin.peek() == '%') fin.ignore(2048, '\n'); // Ignore comments in .mtx file

	fin >> rows_ >> columns_ >> nonZeroValues_;

	matrix_ = new float[rows_*columns_];
	std::fill(matrix_, matrix_ + rows_*columns_, 0.);

	// Read the data
	for (int l = 0; l < nonZeroValues_; l++)
	{
		int row, col;
		float data;
		fin >> row >> col >> data;
		matrix_[(col-1) + (row-1)*rows_] = data;
	}

	fin.close();
}

Matrix::Matrix(float ** D2matrix, int rows, int cols) : // D2Matrix[rows][columns] not the other way
	rows_(rows),
	columns_(cols)
{
	matrix_ = new float[rows_ * columns_];
	nonZeroValues_ = 0;
	for (int i = 0; i < rows_; i++)
	{
		for (int l = 0; l < columns_; l++)
		{
			matrix_[l + i*rows_] = D2matrix[i][l];
			if (D2matrix[i][l] == 0) nonZeroValues_++;
		}
	}
}

Matrix::Matrix(const Matrix &object)
{
	this->rows_ = object.getRows();
	this->columns_ = object.getColumns();
	this->nonZeroValues_ = object.getNonZeroValuesAmount();
	this->matrix_ = new float[rows_ * columns_];

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

float Matrix::getV(int row, int col) const
{
	if (row == 0 || col == 0){
		std::cout << "Matrix::getV - either row or column argument is zero" << std::endl;
		return -1;
	}
	if ((row*col) > (rows_*columns_))
	{
		std::cout << "Matrix::getV - either row or column argument is greater then rows_ or columns_" << std::endl;
		return -1;
	}
	int arrayPos = (row-1)*rows_ + col - 1;
	return matrix_[arrayPos];
}

float * Matrix::getMatrix() const // PB
{
	return this->matrix_;
}

void BasicTests();
void D2MatrixArrayTest();
void Matrix::matrixIntegrationTest() // To replace my main.cpp that was deleted :(
{
	std::cout << "____________________________" << std::endl;
	std::cout << "Basic tests:" << std::endl;
	BasicTests();
	std::cout << "____________________________" << std::endl;
	std::cout << "Testing Matrix constructor for 2D array:" << std::endl;
	D2MatrixArrayTest();
}

void BasicTests()
{
	Matrix x("matrixes/bcsstk03.mtx");
	std::cout << (x.getV(2,1) == float(0));
	std::cout << (x.getV(2,1) == x.getV(1,2));
	std::cout << (x.getRows() == 112);
	std::cout << (x.getColumns() == 112);
	std::cout << (x.getNonZeroValuesAmount() == 376) << std::endl;
	std::cout << "Invalid arguments for getV method:" << std::endl;
	std::cout << (x.getV(0,0) == -1) << std::endl;
	std::cout << (x.getV(122,122) == -1) << std::endl;
}

void D2MatrixArrayTest()
{
	int tmp_row = 2;
	int tmp_col = 2;
	float ** tmp_matrix;
	tmp_matrix = new float*[tmp_row];
	tmp_matrix[0] = new float[tmp_col];
	tmp_matrix[1] = new float[tmp_col];
	tmp_matrix[0][0] = 0;
	tmp_matrix[0][1] = 1;
	tmp_matrix[1][0] = 2;
	tmp_matrix[1][1] = 3;
	Matrix x(tmp_matrix, tmp_row, tmp_col);
	std::cout << (x.getNonZeroValuesAmount() == 1);
	std::cout << (x.getV(1,1) == tmp_matrix[0][0]);
	std::cout << (x.getV(2,2) == tmp_matrix[1][1]) << std::endl;
}