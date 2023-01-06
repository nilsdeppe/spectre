// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"

#include <pup.h>

#include "Parallel/Serialize.hpp"

namespace evolution::dg {
template <size_t Dim>
BoundaryMessage<Dim>::BoundaryMessage(
    const size_t subcell_ghost_data_size_in, const size_t dg_flux_data_size_in,
    const bool sent_across_nodes_in,
    const ::TimeStepId& current_time_step_id_in,
    const ::TimeStepId& next_time_step_id_in,
    const Mesh<Dim>& volume_or_ghost_mesh_in,
    const Mesh<Dim - 1>& interface_mesh_in, double* subcell_ghost_data_in,
    double* dg_flux_data_in)
    : subcell_ghost_data_size(subcell_ghost_data_size_in),
      dg_flux_data_size(dg_flux_data_size_in),
      sent_across_nodes(sent_across_nodes_in),
      current_time_step_id(current_time_step_id_in),
      next_time_step_id(next_time_step_id_in),
      volume_or_ghost_mesh(volume_or_ghost_mesh_in),
      interface_mesh(interface_mesh_in),
      subcell_ghost_data(subcell_ghost_data_in),
      dg_flux_data(dg_flux_data_in) {}

template <size_t Dim>
void* BoundaryMessage<Dim>::pack(BoundaryMessage<Dim>* inmsg) {
  // Size of everything
  size_t totalsize = sizeof(inmsg->subcell_ghost_data_size);
  totalsize += sizeof(inmsg->dg_flux_data_size);
  totalsize += sizeof(inmsg->sent_across_nodes);
  totalsize += sizeof(inmsg->current_time_step_id);
  totalsize += sizeof(inmsg->next_time_step_id);
  totalsize += sizeof(inmsg->volume_or_ghost_mesh);
  totalsize += sizeof(inmsg->interface_mesh);
  totalsize += sizeof(*(inmsg->subcell_ghost_data));
  totalsize += sizeof(*(inmsg->dg_flux_data));

  char* buffer =
      static_cast<char*>(CkAllocBuffer(inmsg, static_cast<int>(totalsize)));

  // First do the size of the array, then the data itself
  PUP::toMem writer(buffer);
  writer | inmsg->subcell_ghost_data_size;
  writer | inmsg->dg_flux_data_size;
  writer | inmsg->sent_across_nodes;
  writer | inmsg->current_time_step_id;
  writer | inmsg->next_time_step_id;
  writer | inmsg->volume_or_ghost_mesh;
  writer | inmsg->interface_mesh;
  if (inmsg->subcell_ghost_data_size != 0) {
    PUParray(writer, inmsg->subcell_ghost_data, inmsg->subcell_ghost_data_size);
  }
  if (inmsg->dg_flux_data_size != 0) {
    PUParray(writer, inmsg->dg_flux_data, inmsg->dg_flux_data_size);
  }

  // Gotta clean up
  delete inmsg;
  return static_cast<void*>(buffer);
}

template <size_t Dim>
BoundaryMessage<Dim>* BoundaryMessage<Dim>::unpack(void* inbuf) {
  // inbuf is the raw memory allocated and assigned in pack. This next buffer is
  // only used to get the sizes of the arrays. It cannot be used to access the
  // data
  BoundaryMessage<Dim>* buffer = reinterpret_cast<BoundaryMessage<Dim>*>(inbuf);

  const size_t subcell_ghost_data_size = buffer->subcell_ghost_data_size;
  const size_t dg_flux_data_size = buffer->dg_flux_data_size;

  BoundaryMessage<Dim>* unpacked_message =
      reinterpret_cast<BoundaryMessage<Dim>*>(
          // FIXME: Is this supposed to be the size of a BoundaryMessage plus
          // the size of the arrays? Or is it supposed to be the size of the
          // inbuf? Because those are two different sizes.
          CkAllocBuffer(inbuf, sizeof(BoundaryMessage<Dim>) +
                                   subcell_ghost_data_size * sizeof(double) +
                                   dg_flux_data_size * sizeof(double)));

  unpacked_message =
      new (static_cast<void*>(unpacked_message)) BoundaryMessage<Dim>();

  PUP::fromMem reader(inbuf);
  reader | unpacked_message->subcell_ghost_data_size;
  reader | unpacked_message->dg_flux_data_size;
  reader | unpacked_message->sent_across_nodes;
  reader | unpacked_message->current_time_step_id;
  reader | unpacked_message->next_time_step_id;
  reader | unpacked_message->volume_or_ghost_mesh;
  reader | unpacked_message->interface_mesh;
  // If we actually have data, set the pointer, then call PUParray to copy the
  // data
  if (subcell_ghost_data_size != 0) {
    unpacked_message->subcell_ghost_data =
        reinterpret_cast<double*>(reinterpret_cast<char*>(unpacked_message) +
                                  sizeof(BoundaryMessage<Dim>));
    PUParray(reader, unpacked_message->subcell_ghost_data,
             subcell_ghost_data_size);
  } else {
    // Otherwise just set the data to a nullptr
    unpacked_message->subcell_ghost_data = nullptr;
  }
  if (dg_flux_data_size != 0) {
    unpacked_message->subcell_ghost_data = reinterpret_cast<double*>(
        reinterpret_cast<char*>(unpacked_message) +
        sizeof(BoundaryMessage<Dim>) + subcell_ghost_data_size);
    PUParray(reader, unpacked_message->dg_flux_data, dg_flux_data_size);
  } else {
    unpacked_message->dg_flux_data = nullptr;
  }

  // Gotta clean up
  CkFreeMsg(inbuf);
  return unpacked_message;
}

template struct BoundaryMessage<1>;
template struct BoundaryMessage<2>;
template struct BoundaryMessage<3>;
}  // namespace evolution::dg
