#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "clockcycle.h"


int main(int argc, char* argv[]){
// ----------------------------------local cpu reduce----------------------------------
  // Initialize the MPI environment
    int world_rank, world_size; // init world rank and size
    MPI_Init(&argc, &argv);

    // Find out rank, size
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // size of a array is determined by how many nodes are working on this task
    int arrSize = (48 * 32**5) / world_size; 

    double* bigArr;
    cudaMallocManaged(&bigArr, arrSize*sizeof(double));

    for (int i=0; i<arrSize; i++){
        bigArr[i] = (double)(i + world_rank * arrSize);
    }
    uint64_t local_reduction_start = clock_now();
    // LOCAL SUM
    double local_sum = 0; 
    for (int i=0; i<arrSize; i++){
        local_sum += bigArr[i];
    }
    uint64_t local_reduction_end = clock_now();



    // calling MPI_Reduce
    double global_sum = 0;
    uint64_t org_start_cycles = clock_now();
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    uint64_t org_end_cycles = clock_now();


    // show runtime
    if (world_rank == 0){
        double p2p_time_in_secs = ((double)(p2p_end_cycles - p2p_start_cycles)) / 512000000;
        printf("MPI Rank %d: Global Sum is %f in %f secs.\n", global_sum, p2p_time_in_secs);
    }

    cudaFree(bigArr);
}
