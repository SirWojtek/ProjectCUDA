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

	matrix_ = new CellInfo[nonZeroValues_];

	// Read the data
	for (int l = 0; l < nonZeroValues_; l++)
	{
		int row, col;
		float data;
		fin >> row >> col >> data;
		matrix_[l].value = data;
		matrix_[l].row = row;
		matrix_[l].column = col;
	}

	fin.close();
}

Matrix::Matrix(CellInfo* inputArray, int rows, int columns, int arraySize) :
	rows_(rows),
	columns_(columns),
	nonZeroValues_(arraySize)
{
	matrix_ = new CellInfo[nonZeroValues_];
	for (int i=0; i<nonZeroValues_; i++)
	{
		matrix_[i].value = inputArray[i].value;
		matrix_[i].row = inputArray[i].row;
		matrix_[i].column = inputArray[i].column;
	}
}

Matrix::Matrix(const Matrix &object)
{
	this->rows_ = object.getRows();
	this->columns_ = object.getColumns();
	this->nonZeroValues_ = object.getNonZeroValuesAmount();
	this->matrix_ = new CellInfo[nonZeroValues_];

	for (int i = 0; i < nonZeroValues_; i++)
	{
		this->matrix_[i] = object.matrix_[i];
	}

}

bool Matrix::operator==(const Matrix &rhs) const
{
	if (this->getColumns() != rhs.getColumns())
		return false;
	if (this->getRows() != rhs.getRows())
		return false;
	if (this->getNonZeroValuesAmount() != rhs.getNonZeroValuesAmount())
		return false;
	for (int i = 0; i < this->getNonZeroValuesAmount(); i++)
	{
		if (this->matrix_[i].value != rhs.matrix_[i].value)
			return false;
		if (this->matrix_[i].row != rhs.matrix_[i].row)
			return false;
		if (this->matrix_[i].column != rhs.matrix_[i].column)
			return false;
	}
	return true;
}

bool Matrix::operator!=(const Matrix &rhs) const
{
	if (*this == rhs)
		return false;
	else
		return true;
}

Matrix& Matrix::operator=(Matrix rhs)
{
	this->swap(*this, rhs);
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
	for (int i=0; i<nonZeroValues_; i++)
	{
		if(( matrix_[i].row == row ) && ( matrix_[i].column == col ) )
		{
			return matrix_[i].value;
		}
	}
	return 0;
}

CellInfo * Matrix::getMatrix() const 
{
	return this->matrix_;
}


int Matrix::countNonZeroValuesAmount(float * inputArray, int arraySize) // Why is this a Matrix member function? - SD
{
	int nonZeroValuesCounter = 0;
	for (int i = 0; i < arraySize; i++)
	{
		if (inputArray[i] != 0.0f)
			nonZeroValuesCounter++;
	}
	return nonZeroValuesCounter;
}

void BasicTests();
void D2MatrixArrayTest();
void Matrix::matrixIntegrationTest() // To replace my main.cpp that was deleted :(
{
	std::cout << "____________________________" << std::endl;
	std::cout << "Basic tests:" << std::endl;
	BasicTests();
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