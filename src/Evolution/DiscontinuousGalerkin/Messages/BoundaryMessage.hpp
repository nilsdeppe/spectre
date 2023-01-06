// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"

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
  Mesh<Dim> volume_or_ghost_mesh{};
  Mesh<Dim - 1> interface_mesh{};

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

  static void* pack(BoundaryMessage*);
  static BoundaryMessage* unpack(void*);
};

// TODO: Add operator== and check that data is same. Add test then using pack
// and unpack
}  // namespace evolution::dg

#define CK_TEMPLATES_ONLY
#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.def.h"
#undef CK_TEMPLATES_ONLY
