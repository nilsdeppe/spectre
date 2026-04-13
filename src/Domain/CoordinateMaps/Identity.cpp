// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Identity.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "Domain/CoordinateMaps/AutodiffInstantiationTypes.hpp"
#include "Utilities/Autodiff/Autodiff.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::CoordinateMaps {

template <size_t Dim>
template <typename T>
void Identity<Dim>::operator()(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, Dim>*> result,
    const std::array<T, Dim>& source_coords) const {
  if constexpr (std::is_same_v<tt::remove_cvref_wrap_t<T>, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(*result, i).destructive_resize(size);
    }
  }
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(*result, i) = dereference_wrapper(gsl::at(source_coords, i));
  }
}

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim> Identity<Dim>::operator()(
    const std::array<T, Dim>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  std::array<ReturnType, Dim> result{};
  (*this)(make_not_null(&result), source_coords);
  return result;
}

template <size_t Dim>
std::optional<std::array<double, Dim>> Identity<Dim>::inverse(
    const std::array<double, Dim>& target_coords) const {
  return make_array<double, Dim>(target_coords);
}

template <size_t Dim>
template <typename T>
void Identity<Dim>::jacobian(
    const gsl::not_null<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>*>
        result,
    const std::array<T, Dim>& source_coords) const {
  if constexpr (std::is_same_v<tt::remove_cvref_wrap_t<T>, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    for (auto& component : *result) {
      component.destructive_resize(size);
    }
  }
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      result->get(i, j) = (i == j) ? 1.0 : 0.0;
    }
  }
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
Identity<Dim>::jacobian(const std::array<T, Dim>& source_coords) const {
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> result{};
  jacobian(make_not_null(&result), source_coords);
  return result;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
Identity<Dim>::inv_jacobian(const std::array<T, Dim>& source_coords) const {
  return identity<Dim>(dereference_wrapper(source_coords[0]));
}

template class Identity<1>;
template class Identity<2>;
template class Identity<3>;

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                           \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)> \
  Identity<DIM(data)>::operator()(                                     \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),   \
                    Frame::NoFrame>                                    \
  Identity<DIM(data)>::jacobian(                                       \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),   \
                    Frame::NoFrame>                                    \
  Identity<DIM(data)>::inv_jacobian(                                   \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;

GENERATE_INSTANTIATIONS(
    INSTANTIATE, (1, 2, 3),
    (double, DataVector,
     std::reference_wrapper<const double>,
     std::reference_wrapper<const DataVector>))

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), MAP_AUTODIFF_TYPES)

#undef INSTANTIATE

#define INSTANTIATE_NOT_NULL(_, data)                                   \
  template void Identity<DIM(data)>::operator()(                        \
      gsl::not_null<                                                    \
          std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>*> \
          result,                                                       \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_NOT_NULL, (1, 2, 3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef INSTANTIATE_NOT_NULL

#define INSTANTIATE_JACOBIAN_NOT_NULL(_, data)                                \
  template void Identity<DIM(data)>::jacobian(                                \
      gsl::not_null<tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data), \
                             Frame::NoFrame>*>                                \
          result,                                                             \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_JACOBIAN_NOT_NULL, (1, 2, 3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef DIM
#undef DTYPE
#undef INSTANTIATE_JACOBIAN_NOT_NULL

}  // namespace domain::CoordinateMaps
