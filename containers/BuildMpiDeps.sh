#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Build the software that is compiled against the MPI in /sxscollaboration/mpi.
# Used by the `mpich` and `openmpi` stages of `containers/Dockerfile.buildenv`
# so both flavors are built the same way. Expects the environment variables
# CHARM_ARCH, PARALLEL_MAKE_ARG and PETSC_VERSION.

set -euo pipefail

: "${CHARM_ARCH:?}" "${PARALLEL_MAKE_ARG:?}" "${PETSC_VERSION:?}"

# Charm++ with the MPI-SMP layer, next to the multicore build from the `dev`
# image and with the same options. Without a compiler option Charm++ compiles
# through the MPI compiler wrappers.
cd /sxscollaboration/charm
./build charm++ "mpi-linux-${CHARM_ARCH}" smp \
  ${PARALLEL_MAKE_ARG} -g -O2 --build-shared --with-production --disable-tls

# PETSc for SpEC, without Fortran
cd /tmp
wget "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz"
tar -xzf "petsc-${PETSC_VERSION}.tar.gz"
cd "petsc-${PETSC_VERSION}"
python ./configure --prefix=/sxscollaboration/petsc \
  --with-mpi-dir=/sxscollaboration/mpi --with-fc=0 \
  --enable-debug=0 --COPTFLAGS=-O3 --CXXOPTFLAGS=-O3 \
  --with-hdf5=0 --download-hdf5=0
make MAKE_NP=4
make install
ldconfig
cd /tmp
rm -r "petsc-${PETSC_VERSION}" "petsc-${PETSC_VERSION}.tar.gz"
