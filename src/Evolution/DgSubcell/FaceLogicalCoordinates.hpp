// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace evolution::dg::subcell::fd {
// @{
/// \ingroup DgSubcellGroup
/// Computes the logical coordinates on the subcell faces.
///
/// The ordering of the coordinates, denoting the lower face in `x` as `Lx` and
/// upper face as `Ux` is (in 2d):
///
/// \code
/// {x_{0,0}^{Lx}, x_{0,0}^{Ux}, x_{1,0}^{Lx}, x_{1,0}^{Ux}...
///  x_{2N,0}^{Lx}, x_{2N,0}^{Ux},
///  x_{0,1}^{Lx}, x_{0,1}^{Ux}, x_{1,1}^{Lx}, x_{1,1}^{Ux}...
///  x_{2N,1}^{Lx}, x_{2N,1}^{Ux}, ...
///  x_{0,2N}^{Lx}, x_{0,2N}^{Ux}, x_{1,2N}^{Lx}, x_{1,2N}^{Ux}...
///  x_{2N,2N}^{Lx}, x_{2N,2N}^{Ux},
///  x_{0,0}^{Ly}, x_{0,0}^{Uy}, x_{1,0}^{Ly}, x_{1,0}^{Uy}...
///  x_{2N,0}^{Ly}, x_{2N,0}^{Uy},
///  x_{0,1}^{Ly}, x_{0,1}^{Uy}, x_{1,1}^{Ly}, x_{1,1}^{Uy}...
///  x_{2N,1}^{Ly}, x_{2N,1}^{Uy}, ...
///  x_{0,2N}^{Ly}, x_{0,2N}^{Uy}, x_{1,2N}^{Ly}, x_{1,2N}^{Uy}...
///  x_{2N,2N}^{Ly}, x_{2N,2N}^{Uy}}
/// \endcode
///
/// That is, the x lower/upper faces are first in x-major ordering, the y
/// lower/upper faces are next in x-major ordering by the cell centers, and
/// finally the z lower/upper faces in x-major ordering by the cell centers.
template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::Logical> face_logical_coordinates(
    const Mesh<VolumeDim>& subcell_mesh) noexcept;

template <size_t Dim>
void face_logical_coordinates(
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Logical>*> logical_x,
    const Mesh<Dim>& subcell_mesh) noexcept;
// @}
}  // namespace evolution::dg::subcell::fd
