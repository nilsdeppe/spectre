#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Build the software that is compiled against the MPI in /sxscollaboration/mpi.
# Used by the `mpich` and `openmpi` stages of `containers/Dockerfile.buildenv`
# so both flavors are built the same way. Expects the environment variables
# CHARM_ARCH, PARALLEL_MAKE_ARG and PETSC_VERSION.

set -euo pipefail

: "${CHARM_ARCH:?}" "${PARALLEL_MAKE_ARG:?}" "${PETSC_VERSION:?}"

# Charm++ and PETSc compile through the MPI compiler wrappers. They must wrap
# GCC, which links with `--as-needed` on Ubuntu, so the MPI C++ bindings
# library isn't recorded as a dependency (see the check at the end).
if [ "$(mpicc -show | cut -d' ' -f1)" != gcc ] \
    || [ "$(mpicxx -show | cut -d' ' -f1)" != g++ ]; then
  echo "Error: The MPI compiler wrappers must wrap gcc and g++:" >&2
  mpicc -show >&2
  mpicxx -show >&2
  exit 1
fi

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

# SpECTRE and SpEC only use the MPI C API, so a host MPI that replaces the
# container's at runtime only has to provide libmpi. Make sure nothing we built
# depends on the MPI C++ bindings library.
for lib in "/sxscollaboration/charm/mpi-linux-${CHARM_ARCH}-smp/lib_so/"*.so \
    /sxscollaboration/petsc/lib/*.so; do
  # Fails the script if the library (or glob match) doesn't exist
  dynamic_section="$(readelf -d "${lib}")"
  if grep -qE 'NEEDED.*lib(mpicxx|mpi_cxx)\.' <<< "${dynamic_section}"; then
    echo "Error: '${lib}' depends on the MPI C++ bindings library." >&2
    exit 1
  fi
done
