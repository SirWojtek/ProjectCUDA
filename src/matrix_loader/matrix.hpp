#pragma once
#include <string>

struct CellInfo
{
public:
	float value;
	int row;
	int column;
};

class Matrix
{
private:
	Matrix() { };
public:
	Matrix(std::string filename);
	Matrix(float *inputArray, int columns, int rows);
	Matrix(CellInfo* inputArray, int rows, int columns, int arraySize);
	Matrix(const Matrix & object); // copy constructor
	~Matrix();

	bool operator==(const Matrix &rhs) const;
	bool operator!=(const Matrix &rhs) const;
	Matrix& operator=(Matrix rhs);
	Matrix& operator+=(const Matrix &rhs);
	friend std::ostream& operator<<(std::ostream& out, const Matrix &obj);
	friend Matrix operator+(Matrix lhs, const Matrix &rhs){ return lhs += rhs; };
	void swap(Matrix& matrix1, Matrix& matrix2);

	int getRows() const;
	int getColumns() const;
	int getNonZeroValuesAmount() const;
	float getV(int row, int col) const;
	CellInfo* getMatrix() const; 
	static void matrixIntegrationTest();

private:
	int rows_;
	int columns_;
	int nonZeroValues_;
	CellInfo* matrix_;

	void addCell(CellInfo &newCell);
	void setV(int row, int col, float value);
	int countNonZeroValuesAmount(float * inputArray, int arraySize); // Why is this a Matrix member function? - SD// it was used by constructor from floats array
};
