#pragma once

#include <string>

class Matrix
{
private:
	Matrix() { };
public:
	Matrix(std::string filename);
	Matrix(const Matrix & object); // copy constructor
	~Matrix();

	Matrix& operator=(Matrix rhs);
	Matrix& operator+=(const Matrix &rhs);
	friend Matrix operator+(Matrix lhs, const Matrix &rhs){ return lhs += rhs; }
	void swap(Matrix& matrix1, Matrix& matrix2);

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
