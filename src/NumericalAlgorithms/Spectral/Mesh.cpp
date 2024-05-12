// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Mesh.hpp"

#include <algorithm>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep
#include <pup_stl.h>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
uint8_t combine(const Spectral::Basis basis,
                const Spectral::Quadrature quadrature) {
  return static_cast<uint8_t>(basis) bitor static_cast<uint8_t>(quadrature);
}
}  // namespace

template <size_t Dim>
Spectral::Basis Mesh<Dim>::extract_basis(const uint8_t bits) {
  return static_cast<Spectral::Basis>(0b11110000 bitand bits);
}

template <size_t Dim>
Spectral::Quadrature Mesh<Dim>::extract_quadrature(const uint8_t bits) {
  return static_cast<Spectral::Quadrature>(0b00001111 bitand bits);
}

template <size_t Dim>
Mesh<Dim>::Mesh(const size_t isotropic_extents, const Spectral::Basis basis,
                const Spectral::Quadrature quadrature) {
  ASSERT(basis != Spectral::Basis::SphericalHarmonic,
         "SphericalHarmonic is not a valid basis for the Mesh");
  ASSERT(isotropic_extents <= 127, "Cannot have more than 127 grid points");
  bit_field_[0] = isotropic_extents;
  bit_field_[3] = combine(basis, quadrature);
  if constexpr (Dim > 1) {
    bit_field_[1] = isotropic_extents;
    bit_field_[4] = combine(basis, quadrature);
    if constexpr (Dim > 2) {
      bit_field_[2] = isotropic_extents;
      bit_field_[5] = combine(basis, quadrature);
    }
  }
}

template <size_t Dim>
Mesh<Dim>::Mesh(const std::array<size_t, Dim>& extents,
                const Spectral::Basis basis,
                const Spectral::Quadrature quadrature) {
  ASSERT(basis != Spectral::Basis::SphericalHarmonic,
         "SphericalHarmonic is not a valid basis for the Mesh");
  if constexpr (Dim > 0) {
    ASSERT(
        extents[0] <= 127,
        "Cannot have more than 127 grid points in direction 0: " << extents[0]);
    bit_field_[0] = extents[0];
    bit_field_[3] = combine(basis, quadrature);
    if constexpr (Dim > 1) {
      ASSERT(extents[1] <= 127,
             "Cannot have more than 127 grid points in direction 1: "
                 << extents[1]);
      bit_field_[1] = extents[1];
      bit_field_[4] = combine(basis, quadrature);
      if constexpr (Dim > 2) {
        bit_field_[2] = extents[2];
        ASSERT(extents[2] <= 127,
               "Cannot have more than 127 grid points in direction 2: "
                   << extents[2]);
        bit_field_[5] = combine(basis, quadrature);
      }
    }
  }
}

template <size_t Dim>
Mesh<Dim>::Mesh(const std::array<size_t, Dim>& extents,
                const std::array<Spectral::Basis, Dim>& bases,
                const std::array<Spectral::Quadrature, Dim>& quadratures) {
  for (auto it = bases.begin(); it != bases.end(); it++) {
    ASSERT(*it != Spectral::Basis::SphericalHarmonic,
           "SphericalHarmonic is not a valid basis for the Mesh");
  }
  if constexpr (Dim > 0) {
    ASSERT(
        extents[0] <= 127,
        "Cannot have more than 127 grid points in direction 0: " << extents[0]);
    bit_field_[0] = extents[0];
    bit_field_[3] = combine(bases[0], quadratures[0]);
    if constexpr (Dim > 1) {
      ASSERT(extents[1] <= 127,
             "Cannot have more than 127 grid points in direction 1: "
                 << extents[1]);
      bit_field_[1] = extents[1];
      bit_field_[4] = combine(bases[1], quadratures[1]);
      if constexpr (Dim > 2) {
        ASSERT(extents[2] <= 127,
               "Cannot have more than 127 grid points in direction 2: "
                   << extents[2]);
        bit_field_[2] = extents[2];
        bit_field_[5] = combine(bases[2], quadratures[2]);
      }
    }
  }
}

template <size_t Dim>
Index<Dim> Mesh<Dim>::extents() const {
  if constexpr (Dim == 0) {
    return Index<Dim>{};
  } else if constexpr (Dim == 1) {
    return Index<Dim>{static_cast<size_t>(bit_field_[0])};
  } else if constexpr (Dim == 2) {
    return Index<Dim>{static_cast<size_t>(bit_field_[0]),
                      static_cast<size_t>(bit_field_[1])};
  } else if constexpr (Dim == 3) {
    return Index<Dim>{static_cast<size_t>(bit_field_[0]),
                      static_cast<size_t>(bit_field_[1]),
                      static_cast<size_t>(bit_field_[2])};
  } else {
    ERROR("Only 0-3d supported in Mesh");
  }
}

template <size_t Dim>
size_t Mesh<Dim>::extents(const size_t d) const {
  ASSERT(d < Dim,
         "Cannot get extent for dim " << d << " in a mesh of Dim " << Dim);
  return gsl::at(bit_field_, d);
}

template <size_t Dim>
size_t Mesh<Dim>::number_of_grid_points() const {
  if constexpr (Dim == 0) {
    return 1;
  } else if constexpr (Dim == 1) {
    return static_cast<size_t>(bit_field_[0]);
  } else if constexpr (Dim == 2) {
    // cast first so we don't overflow
    return static_cast<size_t>(bit_field_[0]) *
           static_cast<size_t>(bit_field_[1]);
  } else if constexpr (Dim == 3) {
    // cast first so we don't overflow
    return static_cast<size_t>(bit_field_[0]) *
           static_cast<size_t>(bit_field_[1]) *
           static_cast<size_t>(bit_field_[2]);
  } else {
    ERROR("Only 0-3d supported in Mesh");
  }
}

template <size_t Dim>
size_t Mesh<Dim>::storage_index(const Index<Dim>& index) const {
  return collapsed_index(index, extents());
}

template <size_t Dim>
std::array<Spectral::Basis, Dim> Mesh<Dim>::basis() const {
  if constexpr (Dim == 0) {
    return {};
  } else if constexpr (Dim == 1) {
    return {extract_basis(bit_field_[3])};
  } else if constexpr (Dim == 2) {
    return {extract_basis(bit_field_[3]), extract_basis(bit_field_[4])};
  } else if constexpr (Dim == 3) {
    return {extract_basis(bit_field_[3]), extract_basis(bit_field_[4]),
            extract_basis(bit_field_[5])};
  } else {
    ERROR("Only 0-3d supported in Mesh");
  }
}

template <size_t Dim>
Spectral::Basis Mesh<Dim>::basis(const size_t d) const {
  return extract_basis(gsl::at(bit_field_, 3 + d));
}

template <size_t Dim>
std::array<Spectral::Quadrature, Dim> Mesh<Dim>::quadrature() const {
  if constexpr (Dim == 0) {
    return {};
  } else if constexpr (Dim == 1) {
    return {extract_quadrature(bit_field_[3])};
  } else if constexpr (Dim == 2) {
    return {extract_quadrature(bit_field_[3]),
            extract_quadrature(bit_field_[4])};
  } else if constexpr (Dim == 3) {
    return {extract_quadrature(bit_field_[3]),
            extract_quadrature(bit_field_[4]),
            extract_quadrature(bit_field_[5])};
  } else {
    ERROR("Only 0-3d supported in Mesh");
  }
}

template <size_t Dim>
Spectral::Quadrature Mesh<Dim>::quadrature(const size_t d) const {
  return extract_quadrature(gsl::at(bit_field_, 3 + d));
}

template <size_t Dim>
// clang-tidy: incorrectly reported redundancy in template expression
template <size_t N, Requires<(N > 0 and N == Dim)>>  // NOLINT
Mesh<Dim - 1> Mesh<Dim>::slice_away(const size_t d) const {
  ASSERT(d < Dim, "Tried to slice away non-existing dimension "
                      << d << " of " << Dim << "-dimensional mesh.");
  std::array<size_t, Dim - 1> dims{};
  for (size_t i = 0; i < d; i++) {
    gsl::at(dims, i) = i;
  }
  for (size_t i = d + 1; i < Dim; i++) {
    gsl::at(dims, i - 1) = i;
  }
  return slice_through(dims);
}

template <size_t Dim>
template <size_t SliceDim, Requires<(SliceDim <= Dim)>>
Mesh<SliceDim> Mesh<Dim>::slice_through(
    const std::array<size_t, SliceDim>& dims) const {
  // Check for duplicates in `dims`
  ASSERT(
      [&dims]() {
        auto sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end());
        auto last_unique = std::unique(sorted_dims.begin(), sorted_dims.end());
        return last_unique == sorted_dims.end();
      }(),
      "Dimensions to slice through contain duplicates.");
  Mesh<SliceDim> result{};
  for (size_t i = 0; i < SliceDim; ++i) {
    const auto& d = gsl::at(dims, i);
    ASSERT(d < Dim, "Tried to slice through non-existing dimension "
                        << d << " of " << Dim << "-dimensional mesh.");
    gsl::at(result.bit_field_, i) = gsl::at(bit_field_, d);
    gsl::at(result.bit_field_, 3 + i) = gsl::at(bit_field_, 3 + d);
  }
  return result;
}

template <size_t Dim>
std::array<Mesh<1>, Dim> Mesh<Dim>::slices() const {
  std::array<Mesh<1>, Dim> result{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = Mesh<1>(extents(d), basis(d), quadrature(d));
  }
  return result;
}

template <size_t Dim>
void Mesh<Dim>::pup(PUP::er& p) {
  p | bit_field_;
}

template <size_t Dim>
bool is_isotropic(const Mesh<Dim>& mesh) {
  if constexpr (Dim == 0 or Dim == 1) {
    return true;
  } else {
    return mesh.extents() == Index<Dim>(mesh.extents(0)) and
           mesh.basis() == make_array<Dim>(mesh.basis(0)) and
           mesh.quadrature() == make_array<Dim>(mesh.quadrature(0));
  }
}

template <size_t Dim>
bool operator==(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs) {
  return lhs.bit_field_ == rhs.bit_field_;
}

template <size_t Dim>
bool operator!=(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const Mesh<Dim>& mesh) {
  using ::operator<<;
  return os << '[' << mesh.extents() << ',' << mesh.basis() << ','
            << mesh.quadrature() << ']';
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GEN_OP(op, dim) \
  template bool operator op(const Mesh<dim>& lhs, const Mesh<dim>& rhs);
#define INSTANTIATE_MESH(_, data)                                 \
  template class Mesh<DIM(data)>;                                 \
  GEN_OP(==, DIM(data))                                           \
  GEN_OP(!=, DIM(data))                                           \
  template std::ostream& operator<<(std::ostream& os,             \
                                    const Mesh<DIM(data)>& mesh); \
  template bool is_isotropic(const Mesh<DIM(data)>& mesh);

#define INSTANTIATE_SLICE_AWAY(_, data) \
  template Mesh<DIM(data) - 1> Mesh<DIM(data)>::slice_away(const size_t) const;
template Mesh<0> Mesh<0>::slice_through(const std::array<size_t, 0>&) const;
template Mesh<0> Mesh<1>::slice_through(const std::array<size_t, 0>&) const;
template Mesh<1> Mesh<1>::slice_through(const std::array<size_t, 1>&) const;
template Mesh<0> Mesh<2>::slice_through(const std::array<size_t, 0>&) const;
template Mesh<1> Mesh<2>::slice_through(const std::array<size_t, 1>&) const;
template Mesh<2> Mesh<2>::slice_through(const std::array<size_t, 2>&) const;
template Mesh<0> Mesh<3>::slice_through(const std::array<size_t, 0>&) const;
template Mesh<1> Mesh<3>::slice_through(const std::array<size_t, 1>&) const;
template Mesh<2> Mesh<3>::slice_through(const std::array<size_t, 2>&) const;
template Mesh<3> Mesh<3>::slice_through(const std::array<size_t, 3>&) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_MESH, (0, 1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE_SLICE_AWAY, (1, 2, 3))

#undef DIM
#undef GEN_OP
#undef INSTANTIATE_MESH
#undef INSTANTIATE_SLICE_AWAY
