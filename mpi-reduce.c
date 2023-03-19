#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "clockcycle.h"



int main(int argc, char* argv[]){
// ----------------------------------MPI_INIT----------------------------------
  // Initialize the MPI environment
    int world_rank, world_size; // init world rank and size
    MPI_Init(&argc, &argv);

    // Find out rank, size
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

// ----------------------------------CUDA_INIT----------------------------------
    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
    printf(" Unable to determine cuda device count, error is %d, count is %d\n",
    cE, cudaDeviceCount );
    exit(-1);
    }
    if( (cE = cudaSetDevice( myrank % cudaDeviceCount )) != cudaSuccess )
    {
    printf(" Unable to have rank %d set to cuda device %d, error is %d \n",
    myrank, (myrank % cudaDeviceCount), cE);
    exit(-1);
    }

    // size of a array is determined by how many nodes are working on this task
    int arrSize = (1<<30) / world_size; 

    int* bigArr = malloc(sizeof(int)*arrSize);

    for (int i=0; i<arrSize; i++){
        bigArr[i] = i + world_rank * arrSize;
    }
 
    // LOCAL SUM
    long long int local_sum = 0; 
    for (int i=0; i<arrSize; i++){
        local_sum += bigArr[i];
    }


    // calling MPI_P2P_Reduce
    long long int global_sum = 0;
    uint64_t p2p_start_cycles = clock_now();
    MPI_P2P_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    uint64_t p2p_end_cycles = clock_now();


    if (world_rank == 0) printf("result from MPI_P2P_Reduce: %lld\n", global_sum);


        // show runtime
    if (world_rank == 0){
        double p2p_time_in_secs = ((double)(p2p_end_cycles - p2p_start_cycles)) / 512000000;
        printf("MPI_P2P_Reduce took %f seconds.\n", p2p_time_in_secs);
    }

    free(bigArr);
    return 0;
}
