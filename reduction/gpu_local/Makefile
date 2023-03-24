all: mpi-reduce-gpu.c cuda-reduce.cu
	mpixlc -O3 mpi-reduce-gpu.c -c -o reduce-mpi-gpu.o
	nvcc -O3 cuda-reduce.cu -c -o reduce-cuda.o
	mpixlc -O3 reduce-mpi-gpu.o reduce-cuda.o -o reduce-exe \
-L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++