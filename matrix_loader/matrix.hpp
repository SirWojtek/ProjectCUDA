#ifndef MATRIX_HPP
#define MATRIX_HPP

class Matrix
{
private:
	Matrix() { };
public:
	Matrix(char* filename);
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

#endif //  MATRIX_HPP