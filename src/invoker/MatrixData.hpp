#pragma once

#include <vector>
#include <utility>

struct MatrixData
{
	std::vector<float> dataVector;
	std::vector<std::pair<unsigned, unsigned> > positionVector;

	float* getRawTable()
	{
		return &dataVector[0];
	}

	void addElement(float data, unsigned row, unsigned column)
	{
		dataVector.push_back(data);
		positionVector.push_back(std::make_pair(row, column));
	}

	void insert(unsigned position, float data, unsigned row, unsigned column)
	{
		dataVector.insert(dataVector.begin() + position, data);
		positionVector.insert(positionVector.begin() + position, std::make_pair(row, column));
	}

	void resize(unsigned newSize)
	{
		dataVector.resize(newSize);
		positionVector.resize(newSize);
	}
};