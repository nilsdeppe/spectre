// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"

#include <pup.h>

#include "Parallel/Serialize.hpp"

namespace evolution::dg {
template <size_t Dim>
void* BoundaryMessage<Dim>::pack(BoundaryMessage* inmsg) {
  // Size of everything
  size_t totalsize = size_of_object_in_bytes(inmsg->subcell_ghost_data_size);
  totalsize += size_of_object_in_bytes(inmsg->dg_flux_data_size);
  totalsize += size_of_object_in_bytes(inmsg->sent_across_nodes);
  totalsize += size_of_object_in_bytes(inmsg->current_time_step_id);
  totalsize += size_of_object_in_bytes(inmsg->next_time_step_id);
  totalsize += size_of_object_in_bytes(inmsg->volume_or_ghost_mesh);
  totalsize += size_of_object_in_bytes(inmsg->interface_mesh);
  totalsize += inmsg->subcell_ghost_data_size * sizeof(double);
  totalsize += inmsg->dg_flux_data_size * sizeof(double);

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
  BoundaryMessage* buffer = static_cast<BoundaryMessage*>(inbuf);

  const size_t subcell_ghost_data_size = buffer->subcell_ghost_data_size;
  const size_t dg_flux_data_size = buffer->dg_flux_data_size;

  BoundaryMessage* unpacked_message = static_cast<BoundaryMessage*>(
      CkAllocBuffer(inbuf, sizeof(BoundaryMessage) +
                               subcell_ghost_data_size * sizeof(double) +
                               dg_flux_data_size * sizeof(double)));

  unpacked_message =
      new (static_cast<void*>(unpacked_message)) BoundaryMessage();

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
  if (unpacked_message->subcell_ghost_data_size != 0) {
    unpacked_message->subcell_ghost_data = reinterpret_cast<double*>(
        reinterpret_cast<char*>(unpacked_message) + sizeof(BoundaryMessage));
    PUParray(reader, unpacked_message->subcell_ghost_data,
             unpacked_message->subcell_ghost_data_size);
  } else {
    // Otherwise just set the data to a nullptr
    unpacked_message->subcell_ghost_data = nullptr;
  }
  if (unpacked_message->dg_flux_data_size != 0) {
    unpacked_message->subcell_ghost_data = reinterpret_cast<double*>(
        reinterpret_cast<char*>(unpacked_message) + sizeof(BoundaryMessage));
    PUParray(reader, unpacked_message->dg_flux_data,
             unpacked_message->dg_flux_data_size);
  } else {
    unpacked_message->dg_flux_data = nullptr;
  }

  // Gotta clean up
  CkFreeMsg(inbuf);
  return unpacked_message;
}
}  // namespace evolution::dg
