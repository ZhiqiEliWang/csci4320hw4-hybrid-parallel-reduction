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

    double* bigArr = malloc(sizeof(int)*arrSize);

    for (int i=0; i<arrSize; i++){
        bigArr[i] = (double)i + world_rank * arrSize;
    }
    uint64_t local_cpu_reduction_start = clock_now();
    // LOCAL SUM
    long long int local_sum = 0; 
    for (int i=0; i<arrSize; i++){
        local_sum += bigArr[i];
    }
    uint64_t local_cpu_reduction_end = clock_now();



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

    free(bigArr);
// ----------------------------------local cuda reduce----------------------------------
  // Initialize the MPI environment

    // Find out rank, size
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // size of a array is determined by how many nodes are working on this task
    arrSize = (1<<30) / world_size; 

    bigArr = malloc(sizeof(int)*arrSize);

    for (int i=0; i<arrSize; i++){
        bigArr[i] = i + world_rank * arrSize;
    }

    // LOCAL SUM
    local_sum = 0; 
    for (int i=0; i<arrSize; i++){
        local_sum += bigArr[i];
    }

    // calling MPI_P2P_Reduce
    global_sum = 0;
    uint64_t org_start_cycles = clock_now();
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    uint64_t org_end_cycles = clock_now();

    MPI_Finalize();

    if (world_rank == 0) printf("result from MPI_Reduce:     %lld\n", global_sum);


        // show runtime
    if (world_rank == 0){
        double org_time_in_secs = ((double)(org_end_cycles - org_start_cycles)) / 512000000;
        printf("MPI_Reduce took     %f seconds.\n", org_time_in_secs);
    }

    free(bigArr);
    return 0;
}
