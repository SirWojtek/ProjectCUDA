ProjectCUDA
===========

###Learning###

Basic materials needed to learn how to CUDA are placed in docs folder. If you want more please visit:

http://docs.nvidia.com/cuda/index.html#axzz3IZNIqBd5

Super Magic Script To List All Cuda Device Properties (SMSTLACDP):
http://www.cs.fsu.edu/~xyuan/cda5125/examples/lect24/devicequery.cu

###Getting Started###

If you want install CUDA compiler (called nvcc) please visit:

#####Linux:#####
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html#axzz3IZQoek6w

#####Windows:#####
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html#axzz3IZQoek6w

###Sparse matrix repository##
http://www.cise.ufl.edu/research/sparse/matrices/

###Overview###

- we try to develop application that should add two sparse matrixes,
- sparse matrixes should be loaded to host memory,
- in host memory matrix shouldn't have field filled with 0,
- before addition in kernel kernel manager prepare matrix data to be added,
- preparation is converting matrix rows that contains non-zero numbers into vector,
- input vectors for kernel is concatenation of non-zero rows from both matrixes,
- if there is non-zero row in one matrix and zero row in second one both rows should be prepared,
- input data for kernel should be two vectors of same size - one for each matrix should be added,
- kernel should add field corresponding to it ids - one kernel add one cell in vector,
- kernel partition: 1 dimension threadId, 1 dimension blockId,
- there is only one computing error appears during calculation,
- error in kernel should be simulating in if statement,
- output from kernel is vector in same size as input,
- kernel manager translate data received from kernel to output matrix,
- error detection should be handled as concurency kernel execution,
- kernel execution must be measured

####Matrix Loader####
- should load matrix from file,
- should not store 0 values in memory,
- should be able to tell what cells contains non-zero values,
- should return non-zero cells in order (from up to down),
- should be able to load matrix from cells vector,

Struct defining single cell:

```struct CellInfo
{
	float value;
	int row;
	int column;
};```

####Kernel####
- should have two implementation - one with error and one without,
- input should be two vectors containing data to add,
- should check if there is need of error injection,
- should inject error (error form is irrelevant),
- should compute index in vector that should be addded basing on threadId.x and blockId.x,
- should make addition - one per each kernel,
- should write addded number into vector

####Kernel Manager####
- should check for non-zero rows in matrix,
- should write non-zero rows into vector,
- should divide task into blocks,
- should call kernel for calculation,
- error detection kernel input can be part of original calculation,
- should call additional kernel calculation for error detection,
- should call error checker to make sure there is no in calculation,
- should write output into matrix

####Error checker####
- should be kernel organized as same as vector addidtion kernel,
- input should be two vectors of same lenght,
- output should be vector of bool of same lenght as input,
- should check for different values in input vectors on same index,
- should set field in output vector to 'false' when detect difference

###Contact###

Please complete contact list below:

GitHub Nick        |        mail address    
-------------------|----------------------------
SirWojtek          |     momatoku@gmail.com 
Gettor             |     sdobroc@gmail.com   
bendzasky          |     bendzasky@gmail.com 
