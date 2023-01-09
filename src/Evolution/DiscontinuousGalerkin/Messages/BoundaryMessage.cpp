// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"

#include <ios>
#include <pup.h>

#include "Parallel/Serialize.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

#include <iostream>
#include <sstream>
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
  // size_t totalsize = sizeof(size_t);
  // // dg_flux_data_size
  // totalsize += sizeof(size_t);
  // // sent_across_nodes
  // totalsize += sizeof(bool);
  // // NOTE: sizeof counts the bool in a TimeStepId as 8 bytes because it's
  // // aligned inside the class, but the PUP::er we use in pack/unpack only
  // // counts
  // // it as 1 byte so we have to subtract off the extra 7 bytes.
  // // current_time_step_id
  // totalsize += sizeof(::TimeStepId) - 7;
  // // next_time_step_id
  // totalsize += sizeof(::TimeStepId) - 7;
  // // volume_or_ghost_mesh
  // totalsize += sizeof(Mesh<Dim>);
  // // QUESTION: For a Mesh<0> the PUP::er calculates the size of the object as
  // // 0 bytes. Not sure why but should we then exclude it from the send
  // entirely?
  // // interface_mesh
  // if constexpr (Dim > 1) {
  //   totalsize += sizeof(Mesh<Dim - 1>);
  // }

  // two sizes
  size_t totalsize = 2 * detail::offset<size_t>();
  // sent_across_nodes
  totalsize += detail::offset<bool>();
  // two TimeStepIds
  totalsize += 2 * detail::offset<TimeStepId>();
  // Mesh<Dim>
  totalsize += detail::offset<Mesh<Dim>>();
  // Mesh<Dim-1>
  totalsize += detail::offset<Mesh<Dim - 1>>();
  // // two double*
  totalsize += 2 * detail::offset<double*>();

  return totalsize;
}

template <size_t Dim>
size_t BoundaryMessage<Dim>::total_size_with_data(const size_t subcell_size,
                                                  const size_t dg_size) {
  size_t totalsize = total_size_without_data();
  totalsize += (subcell_size + dg_size) * sizeof(double);
  return totalsize;
}

template <size_t Dim>
size_t BoundaryMessage<Dim>::total_size_with_data_and_pointers(
    const size_t subcell_size, const size_t dg_size) {
  size_t totalsize = total_size_with_data(subcell_size, dg_size);
  // two double*
  totalsize += 2 * detail::offset<double*>();
  return totalsize;
}

template <size_t Dim>
void* BoundaryMessage<Dim>::pack(BoundaryMessage<Dim>* inmsg) {
  const size_t subcell_size = inmsg->subcell_ghost_data_size;
  const size_t dg_size = inmsg->dg_flux_data_size;

  const size_t totalsize = total_size_with_data(subcell_size, dg_size);

  char* buffer =
      static_cast<char*>(CkAllocBuffer(inmsg, static_cast<int>(totalsize)));
  auto* out_msg = reinterpret_cast<BoundaryMessage*>(buffer);

  memcpy(out_msg, &inmsg->subcell_ghost_data_size,
         BoundaryMessage::total_size_without_data());
  if (subcell_size != 0) {
    out_msg->subcell_ghost_data =
        reinterpret_cast<double*>(std::addressof(out_msg->dg_flux_data)) + 1;
    memcpy(out_msg->subcell_ghost_data, inmsg->subcell_ghost_data,
           subcell_size * sizeof(double));
    Parallel::printf("pack: (%f,%f,%f)\n", inmsg->subcell_ghost_data[0],
                     inmsg->subcell_ghost_data[1],
                     inmsg->subcell_ghost_data[2]);
    Parallel::printf("pack: (%f,%f,%f)\n", out_msg->subcell_ghost_data[0],
                     out_msg->subcell_ghost_data[1],
                     out_msg->subcell_ghost_data[2]);
  }
  if (dg_size != 0) {
    out_msg->dg_flux_data =
        reinterpret_cast<double*>(std::addressof(out_msg->dg_flux_data)) + 1 +
        subcell_size;
    memcpy(out_msg->dg_flux_data, inmsg->dg_flux_data,
           dg_size * sizeof(double));
  }

  delete inmsg;
  return static_cast<void*>(out_msg);
}

template <size_t Dim>
BoundaryMessage<Dim>* BoundaryMessage<Dim>::unpack(void* inbuf) {
  // inbuf is the raw memory allocated and assigned in pack. This next buffer is
  // only used to get the sizes of the arrays. It cannot be used to access the
  // data
  BoundaryMessage<Dim>* buffer = reinterpret_cast<BoundaryMessage<Dim>*>(inbuf);

  const size_t subcell_size = buffer->subcell_ghost_data_size;
  const size_t dg_size = buffer->dg_flux_data_size;

  if (subcell_size != 0) {
    buffer->subcell_ghost_data =
        reinterpret_cast<double*>(std::addressof(buffer->dg_flux_data)) + 1;
    Parallel::printf("unpack: (%f,%f,%f)\n", buffer->subcell_ghost_data[0],
                     buffer->subcell_ghost_data[1],
                     buffer->subcell_ghost_data[2]);
  } else {
    buffer->subcell_ghost_data = nullptr;
  }
  if (dg_size != 0) {
    buffer->dg_flux_data =
        reinterpret_cast<double*>(std::addressof(buffer->dg_flux_data)) + 1 +
        subcell_size;
  } else {
    buffer->dg_flux_data = nullptr;
  }
  return buffer;
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
