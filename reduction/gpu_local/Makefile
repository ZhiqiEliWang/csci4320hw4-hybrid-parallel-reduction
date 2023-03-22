all: reduce-mpi-gpu.c reduce-cuda.cu
<tab> mpixlc -O3 reduce-mpi-gpu.c -c -o reduce-mpi-gpu.o
<tab> nvcc -O3 reduce-cuda.cu -c -o reduce-cuda.o
<tab> mpixlc -O3 reduce-mpi-gpu.o reduce-cuda.o -o reduce-exe \
-L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++