#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "clockcycle.h"


// this function takes the same inputs as MPI_reduce, except MPI_Op is set to MPI_SUM
int MPI_P2P_Reduce(long long int* send_data, // each process's partition of sum
    long long int* recv_data, // the result of reduction, for root only
    int count, // length of the send_data
    MPI_Datatype datatype, // MPI_LONG_LONG in our case
    int root, // 0 in our case
    MPI_Comm communicator) // we only have one communicator
    {

    long long int local_sum = *send_data;
    int self_rank, comm_size;
    MPI_Comm_rank(communicator, &self_rank); // get current rank number
    MPI_Comm_size(communicator, &comm_size); // get total number of nodes
    
    // printf("rank: %d, right before the barrier\n", self_rank);
    MPI_Barrier(communicator); // sync here

    // --------------2. Compute pairwise sums between MPI ranks-------------------
    int stride = 1;
    

    while (stride < comm_size){
        if ((self_rank / stride) % 2 == 1){ // odd ranks after stride: sender
            MPI_Request send_req;
            MPI_Status send_status;
            MPI_Isend(&local_sum, 1, MPI_LONG_LONG, self_rank-stride, 0, communicator, &send_req);
            MPI_Wait(&send_req , MPI_STATUS_IGNORE);
            //printf("rank: %d sent\n", self_rank);
        }
        else{ // even ranks after stride: receiver
            long long int recv_buf;
            MPI_Request recv_req;
            MPI_Status recv_status;
            MPI_Irecv(&recv_buf, 1, MPI_LONG_LONG, self_rank+stride, MPI_ANY_TAG, communicator, &recv_req);
            MPI_Wait(&recv_req , MPI_STATUS_IGNORE);
            //printf("rank: %d received\n", self_rank);
            local_sum += recv_buf; // perform pairwise sum here
        }
        // printf("rank: %d about to hit MPI_Barrierr\n\n", self_rank);
        MPI_Barrier(communicator); // sync here
        stride *= 2;
        //if (self_rank == 0){printf("\n---------stride: %d done-----------\n", stride);}
        // printf("rank: %d just hit barrier\n\n", self_rank);
    }

    if (self_rank == root){*recv_data = local_sum;}
    return 0;
}


int main(int argc, char* argv[]){
// ----------------------------------p2p_REDUCE----------------------------------
  // Initialize the MPI environment
    int world_rank, world_size; // init world rank and size
    MPI_Init(&argc, &argv);

    // Find out rank, size
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
// ----------------------------------MPI_REDUCE----------------------------------
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
