// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/CoordinateMaps/ProductMaps.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <pup.h>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace CoordinateMaps {

namespace product_detail {
template <typename T, size_t Size, typename Map1, typename Map2,
          typename Function, size_t... Is, size_t... Js>
std::array<tt::remove_cvref_wrap_t<T>, Size> apply_call(
    const std::array<T, Size>& coords, const Map1& map1, const Map2& map2,
    const Function func, std::integer_sequence<size_t, Is...> /*meta*/,
    std::integer_sequence<size_t, Js...> /*meta*/) {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  return {
      {func(
           std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Is)>{
               {coords[Is]...}},
           map1)[Is]...,
       func(
           std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Js)>{
               {coords[Map1::dim + Js]...}},
           map2)[Js]...}};
}

template <size_t Size, typename Map1, typename Map2, typename Function,
          size_t... Is, size_t... Js>
std::optional<std::array<double, Size>> apply_inverse(
    const std::array<double, Size>& coords, const Map1& map1, const Map2& map2,
    const Function func, std::integer_sequence<size_t, Is...> /*meta*/,
    std::integer_sequence<size_t, Js...> /*meta*/) {
  auto map1_func =
      func(std::array<double, sizeof...(Is)>{{coords[Is]...}}, map1);
  auto map2_func = func(
      std::array<double, sizeof...(Js)>{{coords[Map1::dim + Js]...}}, map2);
  if (map1_func.has_value() and map2_func.has_value()) {
    return {{{map1_func.value()[Is]..., map2_func.value()[Js]...}}};
  } else {
    return std::nullopt;
  }
}

template <typename T, size_t Size, typename Map1, typename Map2,
          typename Function, size_t... Is, size_t... Js>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Size, Frame::NoFrame> apply_jac(
    const std::array<T, Size>& source_coords, const Map1& map1,
    const Map2& map2, const Function func,
    std::integer_sequence<size_t, Is...> /*meta*/,
    std::integer_sequence<size_t, Js...> /*meta*/) {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  auto map1_jac = func(
      std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Is)>{
          {source_coords[Is]...}},
      map1);
  auto map2_jac = func(
      std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Js)>{
          {source_coords[Map1::dim + Js]...}},
      map2);
  tnsr::Ij<UnwrappedT, Size, Frame::NoFrame> jac{
      make_with_value<UnwrappedT>(dereference_wrapper(source_coords[0]), 0.0)};
  for (size_t i = 0; i < Map1::dim; ++i) {
    for (size_t j = 0; j < Map1::dim; ++j) {
      jac.get(i, j) = std::move(map1_jac.get(i, j));
    }
  }
  for (size_t i = 0; i < Map2::dim; ++i) {
    for (size_t j = 0; j < Map2::dim; ++j) {
      jac.get(Map1::dim + i, Map1::dim + j) = std::move(map2_jac.get(i, j));
    }
  }
  return jac;
}

template <typename T, size_t Size, typename Map1, typename Map2,
          typename Function, size_t... Is, size_t... Js>
void apply_jac_not_null(
    const gsl::not_null<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, Size, Frame::NoFrame>*>
        result,
    const std::array<T, Size>& source_coords, const Map1& map1,
    const Map2& map2, const Function func,
    std::integer_sequence<size_t, Is...> /*meta*/,
    std::integer_sequence<size_t, Js...> /*meta*/) {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  auto map1_jac = func(
      std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Is)>{
          {source_coords[Is]...}},
      map1);
  auto map2_jac = func(
      std::array<std::reference_wrapper<const UnwrappedT>, sizeof...(Js)>{
          {source_coords[Map1::dim + Js]...}},
      map2);
  for (size_t i = 0; i < Map1::dim; ++i) {
    for (size_t j = 0; j < Map1::dim; ++j) {
      result->get(i, j) = std::move(map1_jac.get(i, j));
    }
  }
  for (size_t i = 0; i < Map2::dim; ++i) {
    for (size_t j = 0; j < Map2::dim; ++j) {
      result->get(Map1::dim + i, Map1::dim + j) = std::move(map2_jac.get(i, j));
    }
  }
}
}  // namespace product_detail

template <typename Map1, typename Map2>
ProductOf2Maps<Map1, Map2>::ProductOf2Maps(Map1 map1, Map2 map2)
    : map1_(std::move(map1)),
      map2_(std::move(map2)),
      is_identity_(map1_.is_identity() and map2_.is_identity()) {}

template <typename Map1, typename Map2>
template <typename T>
void ProductOf2Maps<Map1, Map2>::operator()(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, dim>*> result,
    const std::array<T, dim>& source_coords) const {
  if constexpr (std::is_same_v<tt::remove_cvref_wrap_t<T>, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    for (size_t i = 0; i < dim; ++i) {
      (*result)[i].destructive_resize(size);
    }
  }
  *result = product_detail::apply_call(
      source_coords, map1_, map2_,
      [](const auto& point, const auto& map) { return map(point); },
      std::make_index_sequence<Map1::dim>{},
      std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, ProductOf2Maps<Map1, Map2>::dim>
ProductOf2Maps<Map1, Map2>::operator()(
    const std::array<T, dim>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  std::array<ReturnType, dim> result{};
  (*this)(make_not_null(&result), source_coords);
  return result;
}

template <typename Map1, typename Map2>
std::optional<std::array<double, ProductOf2Maps<Map1, Map2>::dim>>
ProductOf2Maps<Map1, Map2>::inverse(
    const std::array<double, dim>& target_coords) const {
  return product_detail::apply_inverse(
      target_coords, map1_, map2_,
      [](const auto& point, const auto& map) {
        return map.inverse(point);
      },
      std::make_index_sequence<Map1::dim>{},
      std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
template <typename T>
void ProductOf2Maps<Map1, Map2>::jacobian(
    const gsl::not_null<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, dim, Frame::NoFrame>*>
        result,
    const std::array<T, dim>& source_coords) const {
  if constexpr (std::is_same_v<tt::remove_cvref_wrap_t<T>, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    for (auto& component : *result) {
      component.destructive_resize(size);
    }
  }
  // Zero all components — only block-diagonal entries are set
  for (auto& component : *result) {
    component = 0.0;
  }
  product_detail::apply_jac_not_null(
      result, source_coords, map1_, map2_,
      [](const auto& point, const auto& map) { return map.jacobian(point); },
      std::make_index_sequence<Map1::dim>{},
      std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, ProductOf2Maps<Map1, Map2>::dim,
         Frame::NoFrame>
ProductOf2Maps<Map1, Map2>::jacobian(
    const std::array<T, dim>& source_coords) const {
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, dim, Frame::NoFrame> result{};
  jacobian(make_not_null(&result), source_coords);
  return result;
}

template <typename Map1, typename Map2>
void ProductOf2Maps<Map1, Map2>::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | map1_;
    p | map2_;
    p | is_identity_;
  }
}

template <typename Map1, typename Map2>
bool operator!=(const ProductOf2Maps<Map1, Map2>& lhs,
                const ProductOf2Maps<Map1, Map2>& rhs) {
  return not(lhs == rhs);
}

template <typename Map1, typename Map2, typename Map3>
ProductOf3Maps<Map1, Map2, Map3>::ProductOf3Maps(Map1 map1, Map2 map2,
                                                 Map3 map3)
    : map1_(std::move(map1)),
      map2_(std::move(map2)),
      map3_(std::move(map3)),
      is_identity_(map1_.is_identity() and map2_.is_identity() and
                   map3_.is_identity()) {}

template <typename Map1, typename Map2, typename Map3>
template <typename T>
void ProductOf3Maps<Map1, Map2, Map3>::operator()(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, dim>*> result,
    const std::array<T, dim>& source_coords) const {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  if constexpr (std::is_same_v<UnwrappedT, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    (*result)[0].destructive_resize(size);
    (*result)[1].destructive_resize(size);
    (*result)[2].destructive_resize(size);
  }
  (*result)[0] =
      map1_(std::array<std::reference_wrapper<const UnwrappedT>, 1>{
          {source_coords[0]}})[0];
  (*result)[1] =
      map2_(std::array<std::reference_wrapper<const UnwrappedT>, 1>{
          {source_coords[1]}})[0];
  (*result)[2] =
      map3_(std::array<std::reference_wrapper<const UnwrappedT>, 1>{
          {source_coords[2]}})[0];
}

template <typename Map1, typename Map2, typename Map3>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, ProductOf3Maps<Map1, Map2, Map3>::dim>
ProductOf3Maps<Map1, Map2, Map3>::operator()(
    const std::array<T, dim>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  std::array<ReturnType, dim> result{};
  (*this)(make_not_null(&result), source_coords);
  return result;
}

template <typename Map1, typename Map2, typename Map3>
std::optional<std::array<double, ProductOf3Maps<Map1, Map2, Map3>::dim>>
ProductOf3Maps<Map1, Map2, Map3>::inverse(
    const std::array<double, dim>& target_coords) const {
  auto c1 = map1_.inverse(std::array<double, 1>{{target_coords[0]}});
  auto c2 = map2_.inverse(std::array<double, 1>{{target_coords[1]}});
  auto c3 = map3_.inverse(std::array<double, 1>{{target_coords[2]}});
  if (c1.has_value() and c2.has_value() and c3.has_value()) {
    return {{{c1.value()[0], c2.value()[0], c3.value()[0]}}};
  } else {
    return std::nullopt;
  }
}

template <typename Map1, typename Map2, typename Map3>
template <typename T>
void ProductOf3Maps<Map1, Map2, Map3>::jacobian(
    const gsl::not_null<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, dim, Frame::NoFrame>*>
        result,
    const std::array<T, dim>& source_coords) const {
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  if constexpr (std::is_same_v<UnwrappedT, DataVector>) {
    const size_t size = dereference_wrapper(source_coords[0]).size();
    for (auto& component : *result) {
      component.destructive_resize(size);
    }
  }
  // Zero all components — only diagonal entries are set
  for (auto& component : *result) {
    component = 0.0;
  }
  get<0, 0>(*result) = get<0, 0>(
      map1_.jacobian(std::array<std::reference_wrapper<const UnwrappedT>, 1>{
          {source_coords[0]}}));
  get<1, 1>(*result) = get<0, 0>(
      map2_.jacobian(std::array<std::reference_wrapper<const UnwrappedT>, 1>{
          {source_coords[1]}}));
  get<2, 2>(*result) = get<0, 0>(
      map3_.jacobian(std::array<std::reference_wrapper<const UnwrappedT>, 1>{
          {source_coords[2]}}));
}

template <typename Map1, typename Map2, typename Map3>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, ProductOf3Maps<Map1, Map2, Map3>::dim,
         Frame::NoFrame>
ProductOf3Maps<Map1, Map2, Map3>::jacobian(
    const std::array<T, dim>& source_coords) const {
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, dim, Frame::NoFrame> result{};
  jacobian(make_not_null(&result), source_coords);
  return result;
}
template <typename Map1, typename Map2, typename Map3>
void ProductOf3Maps<Map1, Map2, Map3>::pup(PUP::er& p) {
  p | map1_;
  p | map2_;
  p | map3_;
  p | is_identity_;
}

template <typename Map1, typename Map2, typename Map3>
bool operator!=(const ProductOf3Maps<Map1, Map2, Map3>& lhs,
                const ProductOf3Maps<Map1, Map2, Map3>& rhs) {
  return not(lhs == rhs);
}
}  // namespace CoordinateMaps
}  // namespace domain
