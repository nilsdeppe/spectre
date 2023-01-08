// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"

#include <ios>
#include <pup.h>

#include "Parallel/Serialize.hpp"
#include "Utilities/GenerateInstantiations.hpp"

#include "Parallel/Printf.hpp"

namespace evolution::dg {
namespace {
void print(const std::string& function, const std::string& message,
           const size_t size) {
  Parallel::printf("%s: %s = %d\n", function, message, size);
}
}  // namespace
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
size_t BoundaryMessage<Dim>::total_size_without_data() {
  // This function must be static to be used in pack and unpack. However, we
  // can't use PUP::sizer or size_of_object_in_bytes in a static member function
  // so we have to use sizeof.

  // subcell_ghost_data_size
  size_t totalsize = sizeof(size_t);
  // dg_flux_data_size
  totalsize += sizeof(size_t);
  // sent_across_nodes
  totalsize += sizeof(bool);
  // NOTE: sizeof counts the bool in a TimeStepId as 8 bytes because it's
  // aligned inside the class, but the PUP::er we use in pack/unpack only counts
  // it as 1 byte so we have to subtract off the extra 7 bytes.
  // current_time_step_id
  totalsize += sizeof(::TimeStepId) - 7;
  // next_time_step_id
  totalsize += sizeof(::TimeStepId) - 7;
  // volume_or_ghost_mesh
  totalsize += sizeof(Mesh<Dim>);
  // QUESTION: For a Mesh<0> the PUP::er calculates the size of the object as 0
  // bytes. Not sure why but should we then exclude it from the send entirely?
  // interface_mesh
  totalsize += sizeof(Mesh<Dim - 1>);

  return totalsize;
}
template <size_t Dim>
void* BoundaryMessage<Dim>::pack(BoundaryMessage<Dim>* inmsg) {
  // Size of everything
  // DEBUG: This is just here so we can print sizes. Will delete later
  size_t totalsize = sizeof(inmsg->subcell_ghost_data_size);
  print("pack", "total size after size_t", totalsize);
  totalsize += sizeof(inmsg->dg_flux_data_size);
  print("pack", "total size after size_t", totalsize);
  totalsize += sizeof(inmsg->sent_across_nodes);
  print("pack", "total size after bool", totalsize);
  totalsize += sizeof(inmsg->current_time_step_id) - 7;
  print("pack", "total size after TimeStepId", totalsize);
  totalsize += sizeof(inmsg->next_time_step_id) - 7;
  print("pack", "total size after TimeStepId", totalsize);
  totalsize += sizeof(inmsg->volume_or_ghost_mesh);
  print("pack", "total size after Mesh<Dim>", totalsize);
  totalsize += sizeof(inmsg->interface_mesh);
  print("pack", "total size after Mesh<Dim-1>", totalsize);
  print("total size without data", "size", total_size_without_data());

  totalsize += inmsg->subcell_ghost_data_size * sizeof(double);
  print("pack", "total size after double*", totalsize);
  totalsize += inmsg->dg_flux_data_size * sizeof(double);
  print("pack", "total size after double*", totalsize);

  totalsize = total_size_without_data() +
              inmsg->subcell_ghost_data_size * sizeof(double) +
              inmsg->dg_flux_data_size * sizeof(double);
  print("pack", "total size", totalsize);

  char* buffer =
      static_cast<char*>(CkAllocBuffer(inmsg, static_cast<int>(totalsize)));

  // First do the size of the array, then the data itself
  PUP::toMem writer(buffer);
  writer | inmsg->subcell_ghost_data_size;
  print("pack", "writer size after size_t", writer.size());
  writer | inmsg->dg_flux_data_size;
  print("pack", "writer size after size_t", writer.size());
  writer | inmsg->sent_across_nodes;
  print("pack", "writer size after bool", writer.size());
  writer | inmsg->current_time_step_id;
  print("pack", "writer size after TimeStepId", writer.size());
  writer | inmsg->next_time_step_id;
  print("pack", "writer size after TimeStepId", writer.size());
  writer | inmsg->volume_or_ghost_mesh;
  print("pack", "writer size after Mesh<Dim>", writer.size());
  writer | inmsg->interface_mesh;
  print("pack", "writer size after Mesh<Dim-1>", writer.size());
  if (inmsg->subcell_ghost_data_size != 0) {
    PUParray(writer, inmsg->subcell_ghost_data, inmsg->subcell_ghost_data_size);
  }
  print("pack", "writer size after double*", writer.size());
  if (inmsg->dg_flux_data_size != 0) {
    PUParray(writer, inmsg->dg_flux_data, inmsg->dg_flux_data_size);
  }
  print("pack", "writer size", writer.size());

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

  print("unpack", "total size without data", total_size_without_data());

  const size_t total_size_with_data = total_size_without_data() +
                                      subcell_ghost_data_size * sizeof(double) +
                                      dg_flux_data_size * sizeof(double);

  print("unpack", "subcell_ghost_data_size",
        subcell_ghost_data_size * sizeof(double));
  print("unpack", "dg_flux_data_size", dg_flux_data_size * sizeof(double));
  print("unpack", "total size with data", total_size_with_data);

  BoundaryMessage<Dim>* unpacked_message =
      reinterpret_cast<BoundaryMessage<Dim>*>(
          CkAllocBuffer(inbuf, total_size_with_data));

  // QUESTION: Is this needed???
  // unpacked_message =
  //     new (static_cast<void*>(unpacked_message)) BoundaryMessage<Dim>();

  PUP::fromMem reader(inbuf);
  reader | unpacked_message->subcell_ghost_data_size;
  Parallel::printf("unpack: unpacked size_t: %d\n",
                   unpacked_message->subcell_ghost_data_size);
  reader | unpacked_message->dg_flux_data_size;
  Parallel::printf("unpack: unpacked size_t: %d\n",
                   unpacked_message->dg_flux_data_size);
  reader | unpacked_message->sent_across_nodes;
  Parallel::printf("unpack: unpacked bool: %s\n",
                   unpacked_message->sent_across_nodes ? "true" : "false");
  reader | unpacked_message->current_time_step_id;
  Parallel::printf("unpack: unpacked TimeStepId: %s\n",
                   unpacked_message->current_time_step_id);
  reader | unpacked_message->next_time_step_id;
  Parallel::printf("unpack: unpacked TimeStepId: %s\n",
                   unpacked_message->next_time_step_id);
  reader | unpacked_message->volume_or_ghost_mesh;
  Parallel::printf("unpack: unpacked Mesh<Dim>: %s\n",
                   unpacked_message->volume_or_ghost_mesh);
  reader | unpacked_message->interface_mesh;
  Parallel::printf("unpack: unpacked Mesh<Dim-1>: %s\n",
                   unpacked_message->interface_mesh);
  // If we actually have data, set the pointer, then call PUParray to copy the
  // data
  if (subcell_ghost_data_size != 0) {
    unpacked_message->subcell_ghost_data = reinterpret_cast<double*>(
        reinterpret_cast<char*>(inbuf) + total_size_without_data());
    PUParray(reader, unpacked_message->subcell_ghost_data,
             subcell_ghost_data_size);
  } else {
    // Otherwise just set the data to a nullptr
    unpacked_message->subcell_ghost_data = nullptr;
  }
  if (dg_flux_data_size != 0) {
    unpacked_message->dg_flux_data = reinterpret_cast<double*>(
        reinterpret_cast<char*>(inbuf) + total_size_without_data() +
        subcell_ghost_data_size * sizeof(double));
    PUParray(reader, unpacked_message->dg_flux_data, dg_flux_data_size);
  } else {
    unpacked_message->dg_flux_data = nullptr;
  }

  // Gotta clean up
  CkFreeMsg(inbuf);
  return unpacked_message;
}

template <size_t Dim>
bool operator==(const BoundaryMessage<Dim>& lhs,
                const BoundaryMessage<Dim>& rhs) {
  const bool equal =
      lhs.subcell_ghost_data_size == rhs.subcell_ghost_data_size and
      lhs.dg_flux_data_size == rhs.dg_flux_data_size and
      lhs.sent_across_nodes == rhs.sent_across_nodes and
      lhs.current_time_step_id == rhs.current_time_step_id and
      lhs.next_time_step_id == rhs.next_time_step_id and
      lhs.volume_or_ghost_mesh == rhs.volume_or_ghost_mesh and
      lhs.interface_mesh == rhs.interface_mesh;

  // We check here first to avoid looping over the arrays if it's unnecessary.
  if (not equal) {
    return false;
  }

  // We are guaranteed that lhs.subcell_size == rhs.subcell_size and
  // lhs.dg_size == rhs.dg_size at this point so it's safe to loop over
  // everything
  for (size_t i = 0; i < lhs.subcell_ghost_data_size; i++) {
    if (lhs.subcell_ghost_data[i] != rhs.subcell_ghost_data[i]) {
      return false;
    }
  }
  for (size_t i = 0; i < lhs.dg_flux_data_size; i++) {
    if (lhs.dg_flux_data[i] != rhs.dg_flux_data[i]) {
      return false;
    }
  }

  return true;
}

template <size_t Dim>
bool operator!=(const BoundaryMessage<Dim>& lhs,
                const BoundaryMessage<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
std::ostream& operator<<(std::ostream& os,
                         const BoundaryMessage<Dim>& message) {
  os << "subcell_ghost_data_size = " << message.subcell_ghost_data_size << "\n";
  os << "dg_flux_data_size = " << message.dg_flux_data_size << "\n";
  os << "sent_across_nodes = " << std::boolalpha << message.sent_across_nodes
     << "\n";
  os << "current_time_ste_id = " << message.current_time_step_id << "\n";
  os << "next_time_ste_id = " << message.next_time_step_id << "\n";
  os << "volume_or_ghost_mesh = " << message.volume_or_ghost_mesh << "\n";
  os << "interface_mesh = " << message.interface_mesh << "\n";

  os << "subcell_ghost_data = (";
  if (message.subcell_ghost_data_size > 0) {
    os << message.subcell_ghost_data[0];
    for (size_t i = 1; i < message.subcell_ghost_data_size; i++) {
      os << "," << message.subcell_ghost_data[i];
    }
  }
  os << ")\n";

  os << "dg_flux_data = (";
  if (message.dg_flux_data_size > 0) {
    os << message.dg_flux_data[0];
    for (size_t i = 1; i < message.dg_flux_data_size; i++) {
      os << "," << message.dg_flux_data[i];
    }
  }
  os << ")";

  return os;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                       \
  template struct BoundaryMessage<DIM(data)>;                      \
  template bool operator==(const BoundaryMessage<DIM(data)>& lhs,  \
                           const BoundaryMessage<DIM(data)>& rhs); \
  template bool operator!=(const BoundaryMessage<DIM(data)>& lhs,  \
                           const BoundaryMessage<DIM(data)>& rhs); \
  template std::ostream& operator<<(                               \
      std::ostream& os, const BoundaryMessage<DIM(data)>& message);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg
