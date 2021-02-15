// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/FaceLogicalCoordinates.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell::fd {
template <size_t Dim>
void face_logical_coordinates(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Logical>*> logical_x,
    const Mesh<Dim>& subcell_mesh) noexcept {
  ASSERT(subcell_mesh.basis() ==
             make_array<Dim>(Spectral::Basis::FiniteDifference),
         "The basis must be finite difference when computing the face "
         "logical coordinates but is "
             << subcell_mesh);
  ASSERT(subcell_mesh.quadrature() ==
             make_array<Dim>(Spectral::Quadrature::CellCentered),
         "The quadrature of the mesh must be CellCentered when computing the "
         "face logical coordinates but is "
             << subcell_mesh);

  size_t total_face_points = (subcell_mesh.extents(0) + 1) *
                             subcell_mesh.slice_away(0).number_of_grid_points();
  for (size_t d = 1; d < Dim; ++d) {
    total_face_points += (subcell_mesh.extents(d) + 1) *
                         subcell_mesh.slice_away(d).number_of_grid_points();
  }

  destructive_resize_components(logical_x, total_face_points);
  for (size_t d = 0; d < Dim; ++d) {
    for (size_t face_grid_index = 0, offset = 0; face_grid_index < Dim;
         ++face_grid_index,
                offset += (subcell_mesh.extents(d) + 1) *
                          subcell_mesh.slice_away(d).number_of_grid_points()) {
      const DataVector& collocation_points_in_this_dim =
          d == face_grid_index ? Spectral::collocation_points<
                                     Spectral::Basis::FiniteDifference,
                                     Spectral::Quadrature::FaceCentered>(
                                     subcell_mesh.extents(face_grid_index) + 1)
                               : Spectral::collocation_points<
                                     Spectral::Basis::FiniteDifference,
                                     Spectral::Quadrature::CellCentered>(
                                     subcell_mesh.extents(face_grid_index));
      Index<Dim> extents = subcell_mesh.extents();
      extents[face_grid_index] += 1;
      for (IndexIterator<Dim> index(extents); index; ++index) {
        logical_x->get(d)[offset + index.collapsed_index()] =
            collocation_points_in_this_dim[index()[d]];
      }
    }
  }
}

template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::Logical> face_logical_coordinates(
    const Mesh<Dim>& subcell_mesh) noexcept {
  tnsr::I<DataVector, Dim, Frame::Logical> logical_x{};
  face_logical_coordinates(make_not_null(&logical_x), subcell_mesh);
  return logical_x;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                            \
  template tnsr::I<DataVector, DIM(data), Frame::Logical>                 \
  face_logical_coordinates(const Mesh<DIM(data)>& subcell_mesh) noexcept; \
  template void face_logical_coordinates(                                 \
      gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Logical>*>,     \
      const Mesh<DIM(data)>& subcell_mesh) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::fd
