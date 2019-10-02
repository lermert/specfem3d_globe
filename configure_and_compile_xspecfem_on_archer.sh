#! /bin/bash -l

# Clean up anything from before
rm -rf OUTPUT_FILES/*
rm -rf bin
rm -rf bin.forward
rm -rf bin.kernel


module load PrgEnv-cray
module load cray-mpich
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel
module load boost

FC=ftn CC=cc MPIF90=ftn MPICC=cc \
MPI_INC=$CRAY_MPICH_DIR/include \
FCLIBS=" " \
FLAGS_CHECK='-O3' \
CFLAGS='-O3' \
FCFLAGS='-O3' \
./configure

mkdir -p bin
make xmeshfem3D -j 4
make xspecfem3D -j 4

