all: compile

compile: main.o matrix.o
	nvcc main.o matrix.o -o out

main.o: main.cu
	nvcc -c main.cu

matrix.o: matrix.cpp
	nvcc -c matrix.cpp

run:
	./out

clean:
	rm -rf *.o out