#pragma once

#include <string>

class Matrix
{
private:
	Matrix() { };
public:
	Matrix(std::string filename);
	~Matrix();
	int getRows() const;
	int getColumns() const;
	int getNonZeroValuesAmount() const;
	double getV(int row, int col) const;
	double * getMatrix() const; // PB - lazy solution
private:
	int rows_;
	int columns_;
	int nonZeroValues_;
	double* matrix_;
};
