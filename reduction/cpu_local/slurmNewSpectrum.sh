#!/bin/bash -x

module load spectrum-mpi

#####################################################################################################
# Launch N tasks per compute node allocated. Per below this launches 32 MPI rank per compute node.
# taskset insures that hyperthreaded cores are skipped.
#####################################################################################################
# taskset -c 0-159:4 mpirun -N 32 /gpfs/u/home/SPNR/SPNRcaro/scratch/MPI-Examples/mpi-hello
mpicc mpi-reduce-cpu.c -o mpi-reduce-cpu
taskset -c 0-159:4 mpirun -N 32 /gpfs/u/home/PCPC/PCPCwnww/scratch/csci4320hw4-hybrid-parallel-reduction/reduction/cpu_local/mpi-reduce-cpu
