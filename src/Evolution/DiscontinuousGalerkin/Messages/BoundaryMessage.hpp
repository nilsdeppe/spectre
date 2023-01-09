// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>
#include <type_traits>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/PrettyType.hpp"

#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.decl.h"

namespace evolution::dg {
template <size_t Dim>
struct BoundaryMessage : public CMessage_BoundaryMessage<Dim> {
  using base = CMessage_BoundaryMessage<Dim>;
  // TODO: Put tag here inside message because they are tied together

  size_t subcell_ghost_data_size;
  size_t dg_flux_data_size;
  bool sent_across_nodes;
  ::TimeStepId current_time_step_id;
  ::TimeStepId next_time_step_id;
  Mesh<Dim> volume_or_ghost_mesh;
  Mesh<Dim - 1> interface_mesh;

  // If set to nullptr then we aren't sending that type of data.
  double* subcell_ghost_data;
  double* dg_flux_data;

  BoundaryMessage() = default;

  BoundaryMessage(const size_t subcell_ghost_data_size_in,
                  const size_t dg_flux_data_size_in,
                  const bool sent_across_nodes_in,
                  const ::TimeStepId& current_time_step_id_in,
                  const ::TimeStepId& next_time_step_id_in,
                  const Mesh<Dim>& volume_or_ghost_mesh_in,
                  const Mesh<Dim - 1>& interface_mesh_in,
                  double* subcell_ghost_data_in, double* dg_flux_data_in);

  static size_t total_size_without_data();
  static size_t total_size_with_data(const size_t subcell_size,
                                     const size_t dg_size);
  static size_t total_size_with_data_and_pointers(const size_t subcell_size,
                                                  const size_t dg_size);

  static void* pack(BoundaryMessage*);
  static BoundaryMessage* unpack(void*);
};

template <size_t Dim>
bool operator==(const BoundaryMessage<Dim>& lhs,
                const BoundaryMessage<Dim>& rhs);
template <size_t Dim>
bool operator!=(const BoundaryMessage<Dim>& lhs,
                const BoundaryMessage<Dim>& rhs);

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const BoundaryMessage<Dim>& message);

namespace detail {
template <typename T>
size_t offset() {
  if constexpr (std::is_same_v<T, size_t>) {
    return sizeof(size_t);
  } else if constexpr (std::is_same_v<T, bool>) {
    // BoundaryMessage is 8-byte aligned so a bool isn't 1 byte, it's actually 8
    return 8;
  } else if constexpr (std::is_same_v<T, TimeStepId>) {
    return sizeof(TimeStepId);
  } else if constexpr (std::is_same_v<T, Mesh<3>> or
                       std::is_same_v<T, Mesh<2>> or
                       std::is_same_v<T, Mesh<1>>) {
    return sizeof(T);
  } else if constexpr (std::is_same_v<T, Mesh<0>>) {
    // Mesh<0> is only 3 bytes, but we need 8 for alignment
    return 8;
  } else if constexpr (std::is_same_v<T, double*>) {
    return sizeof(double*);
  } else {
    ERROR("Cannot calculate offset for '"
          << pretty_type::name<T>()
          << "' in a BoundaryMessage. Offset is only known for size_t, bool, "
             "TimeStepId, Mesh, double*.");
    return 0;
  }
}
}  // namespace detail

// TODO: Add operator== and check that data is same. Add test then using pack
// and unpack
}  // namespace evolution::dg

#define CK_TEMPLATES_ONLY
#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.def.h"
#undef CK_TEMPLATES_ONLY
