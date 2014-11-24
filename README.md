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

1. Kernel implementation
    - kernel should make addition of two elements, position of added elements depends on kernel index,
    - kernel must be able to inject error, there is possibility of multiple error across kernels.
2. Kernel manager implementation.
    - kernel manager task is to divide task to kernels, basing on input data size,
    - kernel manager should have functionality to trigger additional redundancy computing to check for errors,
    - if km find an error, it should compute value again (with another error check?).
3. Matrix loading.
    - loading from file,
    - format can be chosen from one from repository site,
    - loaded matrix should be placed in 2D matrix in host memory.

Please assign yourself to one task. I suggest to start coding in newly created file SparseMatrix.cu.
**Take notice, that we should have basic implementation finished until 28.11.2014.**
If there is any question, you are free to contact me:)

Person       |     Task
-------------|----------------
SirWojtek    | 2
Gettor       | 3
bendzasky    | 1


###Contact###

Please complete contact list below:

GitHub Nick        |        mail address     |    telephone
-------------------|-------------------------|----------------------
SirWojtek          |     momatoku@gmail.com  |   781 842 090
Gettor             |    sdobroc@gmail.com    |   601 553 817
bendzasky          |    bendzasky@gmail.com  |   509 409 563  
