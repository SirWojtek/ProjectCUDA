#pragma once

#include <string>

class Matrix
{
private:
	Matrix() { };
public:
	Matrix(std::string filename);
	Matrix(float ** D2matrix, int rows, int cols);
	Matrix(const Matrix & object); // copy constructor
	~Matrix();

	Matrix& operator=(Matrix rhs);
	Matrix& operator+=(const Matrix &rhs);
	friend Matrix operator+(Matrix lhs, const Matrix &rhs){ return lhs += rhs; }
	void swap(Matrix& matrix1, Matrix& matrix2);

	int getRows() const;
	int getColumns() const;
	int getNonZeroValuesAmount() const;
	float getV(int row, int col) const;
	float * getMatrix() const; // PB - lazy solution
	static void matrixIntegrationTest();
private:
	int rows_;
	int columns_;
	int nonZeroValues_;
	float* matrix_;
};
