#include <stdio.h>

template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    mySum += __shfl_down_sync(mask, mySum, offset);
  }
  return mySum;
}

#if __CUDA_ARCH__ >= 800
// Specialize warpReduceFunc for int inputs to use __reduce_add_sync intrinsic
// when on SM 8.0 or higher
template <>
__device__ __forceinline__ int warpReduceSum<int>(unsigned int mask,
                                                  int mySum) {
  mySum = __reduce_add_sync(mask, mySum);
  return mySum;
}
#endif

// reduce7 is ported from CUDA sample code
template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce7(const T *__restrict__ g_idata, T *__restrict__ g_odata,
                        unsigned int n) {
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;
  unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
  maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
  const unsigned int mask = (0xffffffff) >> maskLength;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum += g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum += g_idata[i];
      i += gridSize;
    }
  }

  // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
  // SM 8.0
  mySum = warpReduceSum<T>(mask, mySum);

  // each thread puts its local sum into shared memory
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = mySum;
  }

  __syncthreads();

  const unsigned int shmem_extent =
      (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
  const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
  if (tid < shmem_extent) {
    mySum = sdata[tid];
    // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum<T>(ballot_result, mySum);
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = mySum;
  }
}


extern "C"
void arrInit(double* bigArr, int arrSize, int rank){
  cudaMallocManaged(&bigArr, sizeof(double)*arrSize);
  for (int i=0; i<arrSize; i++){
    bigArr[i] = (double)(i + rank * arrSize);
  }
}


// a helper to check power of 2
extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

extern "C"
void cudaReduce(double* input, double* output, int size) {
  cudaMallocManaged(&output, sizeof(double));
  int dimBlock = 1024; // we hardcode block size as 1024
  int dimGrid = (size + dimBlock - 1) / dimBlock;
  int smemSize = ((1024 / 32) + 1) * sizeof(double);
  bool isPow = isPow2(size);
  printf("Rank %d: reduction started");
  reduce7<double, 1024, false><<<dimGrid, dimBlock, smemSize>>>(input, output, size);
  return;
}

extern "C"
void freeCudaMem(double* ptr){
  cudaFree(ptr);
}

extern "C"
void cudaInit(int world_rank){
  int cE;
  int cudaDeviceCount;
  if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
  {
  printf(" Unable to determine cuda device count, error is %d, count is %d\n",
  cE, cudaDeviceCount );
  exit(-1);
  }
  if( (cE = cudaSetDevice( world_rank % cudaDeviceCount )) != cudaSuccess )
  {
  printf(" Unable to have rank %d set to cuda device %d, error is %d \n",
  world_rank, (world_rank % cudaDeviceCount), cE);
  exit(-1);
  }
}

