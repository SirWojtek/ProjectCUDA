ProjectCUDA
===========

###Learning###

Basic materials needed to learn how to CUDA are placed in docs folder. If you want more please visit:

http://docs.nvidia.com/cuda/index.html#axzz3IZNIqBd5

###Getting Started###

If you want install CUDA compiler (called nvcc) please visit:

#####Linux:#####
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html#axzz3IZQoek6w

#####Windows:#####
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html#axzz3IZQoek6w

###Sparse matrix repository##
http://www.cise.ufl.edu/research/sparse/matrices/

###Tasks###

####Already done####

Kernel implementation **done**
    * kernel should make addition of two elements, position of added elements depends on kernel index,
    * kernel must be able to inject error, there is possibility of multiple error across kernels.


Matrix loading **completed. Matrix chosen: http://www.cise.ufl.edu/research/sparse/matrices/HB/bcsstk03.html**
    * loading from file,
    * format can be chosen from one from repository site,
    * loaded matrix should be placed in 2D matrix in host memory.


####To do####

1. Kernel manager implementation
    - kernel manager task is to divide task to kernels, basing on input data size,
    - kernel manager should have functionality to trigger additional redundancy computing to check for errors,
2. Time measurement
	- how to measure kernel execution time?
	- implementation of time measurement tool
3. Error recognition
	- assume you have result matrix computed by CUDA,
	- check for errors that may appear while computing


Please assign yourself to one task.
**Next deadline: 05.12.2014**
If there is any question, you are free to contact me:)

Person       |     Task
-------------|----------------
SirWojtek    | 1
Gettor       | -
bendzasky    | -


###Contact###

Please complete contact list below:

GitHub Nick        |        mail address     |    telephone
-------------------|-------------------------|----------------------
SirWojtek          |     momatoku@gmail.com  |   781 842 090
Gettor             |    sdobroc@gmail.com    |   601 553 817
bendzasky          |    bendzasky@gmail.com  |   509 409 563  
