#pragma once

#include <string>

class Matrix
{
private:
	Matrix() { };
public:
	Matrix(std::string filename);
	~Matrix();
	int getRows();
	int getColumns();
	int getNonZeroValuesAmount();
	double getV(int row, int col);
private:
	int rows_;
	int columns_;
	int nonZeroValues_;
	double* matrix_;
};
