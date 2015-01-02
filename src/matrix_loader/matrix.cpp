#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

#include <stdlib.h>
#include <windows.h>
#include <time.h>

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

Matrix& Matrix::operator+=(const Matrix &rhs)
{
	CellInfo * thisMatrix = this->getMatrix();
	CellInfo * addedMatrix = rhs.getMatrix();
	int vectorSize = rhs.getNonZeroValuesAmount();
	int row = 0, column = 0;
	float actualValue = 0, newValue = 0;

	for (int i = 0; i < vectorSize; i++)
	{
		row = addedMatrix[i].row;
		column = addedMatrix[i].column;
		actualValue = this->getV(row, column);
		newValue = actualValue + addedMatrix[i].value;
		if (actualValue == 0)
		{
			CellInfo newCell = {newValue, row, column};
			this->addCell(newCell);
		}
		else
		{
			this->setV(row, column, newValue);
		}
	}

	return *this;
}

std::ostream& operator<<(std::ostream& out, const Matrix &obj)
{
	out << "Pretty Print starting:" << std::endl;
	out << "Rows:" << obj.rows_ << std::endl;
	out << "Columns:" << obj.columns_ << std::endl;
	out << "Non Zero Values:" << obj.nonZeroValues_ << std::endl;
	out << "Values (row/column/value):" << std::endl;
	for (int i=0; i<obj.nonZeroValues_; i++)
	{
		out << obj.matrix_[i].row << " " << obj.matrix_[i].column << " " << obj.matrix_[i].value << std::endl;
	}
	out << "Pretty Print ended." << std::endl;
	return out;
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

Matrix * Matrix::randomize() const
{
	Matrix* randomizedMatrix = new Matrix();
	randomizedMatrix->nonZeroValues_ = this->nonZeroValues_;
	randomizedMatrix->columns_ = this->columns_;
	randomizedMatrix->rows_ = this->rows_;
	randomizedMatrix->matrix_ = new CellInfo[randomizedMatrix->nonZeroValues_];
	int* range = new int[this->nonZeroValues_];
	for (int i=0; i<this->nonZeroValues_; i++)
	{
		range[i] = i;
	}
	srand (time(NULL));
	for (int i=0; i<(this->nonZeroValues_ * this->nonZeroValues_); i++)
	{
		int a = rand() % this->nonZeroValues_;
		int b = rand() % this->nonZeroValues_;
		if (a == b )
		{
			if (a != (this->nonZeroValues_ - 1)) a++;
			else a--;
		}
		int tmp = range[a];
		range[a] = range[b];
		range[b] = tmp;
	}
	for (int i=0; i<this->nonZeroValues_; i++)
	{
		randomizedMatrix->matrix_[i] = this->matrix_[range[i]];
	}
	delete[] range;
	return randomizedMatrix;
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
	std::cout << (x.getV(0,0) == 0) << std::endl;
	std::cout << (x.getV(122,122) == 0) << std::endl;
	// std::cout << "Printing whole matrix:" << std::endl;
	// std::cout << x;
	std::cout << "Randomizing matrix:" << std::endl;
	Matrix * y = x.randomize();
	// std::cout << "Printing randomized matrix:" << std::endl;
	// std::cout << *y;
	std::cout << "Checking if randomize actually did its job:" << std::endl;
	std::cout << (*y != x) << std::endl;
}


void Matrix::addCell(CellInfo &newCell)
{
	int oldSize = getNonZeroValuesAmount();
	int newSize = oldSize + 1;
	CellInfo *temp = new CellInfo[newSize];
	std::copy(matrix_, matrix_ + oldSize, temp);
	delete[] matrix_;
	matrix_ = temp;
	matrix_[oldSize] = newCell; // 
	nonZeroValues_++;
}

void Matrix::setV(int row, int col, float value)
{
	for (int i = 0; i<nonZeroValues_; i++)
	{
		if ((matrix_[i].row == row) && (matrix_[i].column == col))
		{
			matrix_[i].value = value;
		}
	}
}