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
  // totalsize += 2 * detail::offset<double*>();

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
  // Size of everything
  // DEBUG: This is just here so we can print sizes. Will delete later
  // size_t totalsize = sizeof(inmsg->subcell_ghost_data_size);
  // print("pack", "total size after size_t", totalsize);
  // totalsize += sizeof(inmsg->dg_flux_data_size);
  // print("pack", "total size after size_t", totalsize);
  // totalsize += sizeof(inmsg->sent_across_nodes);
  // print("pack", "total size after bool", totalsize);
  // totalsize += sizeof(inmsg->current_time_step_id) - 7;
  // print("pack", "total size after TimeStepId", totalsize);
  // totalsize += sizeof(inmsg->next_time_step_id) - 7;
  // print("pack", "total size after TimeStepId", totalsize);
  // totalsize += sizeof(inmsg->volume_or_ghost_mesh);
  // // print("pack", "total size after Mesh<Dim>", totalsize);
  // Parallel::printf("pack: total size after Mesh<%d> = %d\n", Dim, totalsize);
  // if constexpr (Dim > 1) {
  //   totalsize += sizeof(inmsg->interface_mesh);
  // }
  // // print("pack", "total size after Mesh<Dim-1>", totalsize);
  // Parallel::printf("pack: total size after Mesh<%d> = %d\n", Dim - 1,
  //                  totalsize);
  // print("total size without data", "size", total_size_without_data());

  // totalsize += inmsg->subcell_ghost_data_size * sizeof(double);
  // print("pack", "total size after double*", totalsize);
  // totalsize += inmsg->dg_flux_data_size * sizeof(double);
  // print("pack", "total size after double*", totalsize);

  // totalsize = total_size_without_data() +
  //             inmsg->subcell_ghost_data_size * sizeof(double) +
  //             inmsg->dg_flux_data_size * sizeof(double);
  // print("pack", "total size", totalsize);

  // char* buffer =
  //     static_cast<char*>(CkAllocBuffer(inmsg, static_cast<int>(totalsize)));

  // // First do the size of the array, then the data itself
  // PUP::toMem writer(buffer);
  // writer | inmsg->subcell_ghost_data_size;
  // print("pack", "writer size after size_t", writer.size());
  // writer | inmsg->dg_flux_data_size;
  // print("pack", "writer size after size_t", writer.size());
  // writer | inmsg->sent_across_nodes;
  // print("pack", "writer size after bool", writer.size());
  // writer | inmsg->current_time_step_id;
  // print("pack", "writer size after TimeStepId", writer.size());
  // writer | inmsg->next_time_step_id;
  // print("pack", "writer size after TimeStepId", writer.size());
  // writer | inmsg->volume_or_ghost_mesh;
  // // print("pack", "writer size after Mesh<Dim>", writer.size());
  // Parallel::printf("pack: total size after Mesh<%d> = %d\n", Dim,
  //                  writer.size());
  // if constexpr (Dim > 1) {
  //   writer | inmsg->interface_mesh;
  // }
  // // print("pack", "writer size after Mesh<Dim-1>", writer.size());
  // Parallel::printf("pack: total size after Mesh<%d> = %d\n", Dim - 1,
  //                  writer.size());
  // if (inmsg->subcell_ghost_data_size != 0) {
  //   PUParray(writer, inmsg->subcell_ghost_data,
  //   inmsg->subcell_ghost_data_size);
  // }
  // print("pack", "writer size after double*", writer.size());
  // if (inmsg->dg_flux_data_size != 0) {
  //   PUParray(writer, inmsg->dg_flux_data, inmsg->dg_flux_data_size);
  // }
  // print("pack", "writer size", writer.size());

  // Gotta clean up
  // delete inmsg;
  // return static_cast<void*>(buffer);

  // DEBUG: memcpy approach... not working
  const size_t subcell_size = inmsg->subcell_ghost_data_size;
  const size_t dg_size = inmsg->dg_flux_data_size;

  const size_t totalsize = total_size_with_data(subcell_size, dg_size);

  char* buffer =
      static_cast<char*>(CkAllocBuffer(inmsg, static_cast<int>(totalsize)));
  char* original_buffer = buffer;

  memcpy(buffer, &inmsg->subcell_ghost_data_size, detail::offset<size_t>());
  buffer += detail::offset<size_t>();
  memcpy(buffer, &inmsg->dg_flux_data_size, detail::offset<size_t>());
  buffer += detail::offset<size_t>();
  memcpy(buffer, &inmsg->sent_across_nodes, detail::offset<bool>());
  buffer += detail::offset<bool>();
  memcpy(buffer, &inmsg->current_time_step_id, detail::offset<TimeStepId>());
  buffer += detail::offset<TimeStepId>();
  memcpy(buffer, &inmsg->next_time_step_id, detail::offset<TimeStepId>());
  buffer += detail::offset<TimeStepId>();
  memcpy(buffer, &inmsg->volume_or_ghost_mesh, detail::offset<Mesh<Dim>>());
  buffer += detail::offset<Mesh<Dim>>();
  memcpy(buffer, &inmsg->interface_mesh, detail::offset<Mesh<Dim - 1>>());
  buffer += detail::offset<Mesh<Dim - 1>>();
  memcpy(buffer, &inmsg->subcell_ghost_data, detail::offset<double*>());
  buffer += detail::offset<double*>();
  memcpy(buffer, &inmsg->dg_flux_data, detail::offset<double*>());
  buffer += detail::offset<double*>();
  if (subcell_size != 0) {
    memcpy(buffer, inmsg->subcell_ghost_data, subcell_size * sizeof(double));
    Parallel::printf("pack: (%f,%f,%f)\n",
    buffer[0], buffer[1], buffer[2]
    );
    buffer += subcell_size;
  }
  if (dg_size != 0) {
    memcpy(buffer, inmsg->dg_flux_data, dg_size * sizeof(double));
    buffer += dg_size;
  }

  delete inmsg;
  return static_cast<void*>(original_buffer);
}

template <size_t Dim>
BoundaryMessage<Dim>* BoundaryMessage<Dim>::unpack(void* inbuf) {
  // inbuf is the raw memory allocated and assigned in pack. This next buffer is
  // only used to get the sizes of the arrays. It cannot be used to access the
  // data
  // BoundaryMessage<Dim>* buffer =
  // reinterpret_cast<BoundaryMessage<Dim>*>(inbuf);

  // const size_t subcell_size = buffer->subcell_ghost_data_size;
  // const size_t dg_size = buffer->dg_flux_data_size;

  // print("unpack", "total size without data", total_size_without_data());

  // const size_t total_size_with_data = total_size_without_data() +
  //                                     subcell_size * sizeof(double) +
  //                                     dg_size * sizeof(double);

  // print("unpack", "subcell_size", subcell_size * sizeof(double));
  // print("unpack", "dg_size", dg_size * sizeof(double));
  // print("unpack", "total size with data", total_size_with_data);

  // BoundaryMessage<Dim>* unpacked_message =
  //     reinterpret_cast<BoundaryMessage<Dim>*>(
  //         CkAllocBuffer(inbuf, total_size_with_data));

  // // QUESTION: Is this needed???
  // // unpacked_message =
  // //     new (static_cast<void*>(unpacked_message)) BoundaryMessage<Dim>();

  // Parallel::printf("unpack: sizeof BoundaryMessage: %d\n",
  //                  sizeof(BoundaryMessage<Dim>));
  // Parallel::printf("unpack: sizeof unpacked_message: %d\n",
  //                  sizeof(*unpacked_message));

  // PUP::fromMem reader(inbuf);
  // size_t prev_reader_size = reader.size();
  // Parallel::printf("unpack: reader size: %d\n", prev_reader_size);
  // reader | unpacked_message->subcell_ghost_data_size;
  // size_t obj_size = reader.size() - prev_reader_size;
  // Parallel::printf(
  //     "unpack: unpacked size_t: %d, size_t size: %d, reader size: %d\n",
  //     unpacked_message->subcell_ghost_data_size, obj_size, reader.size());
  // prev_reader_size = reader.size();
  // reader | unpacked_message->dg_flux_data_size;
  // obj_size = reader.size() - prev_reader_size;
  // Parallel::printf(
  //     "unpack: unpacked size_t: %d, size_t size: %d, reader size: %d\n",
  //     unpacked_message->dg_flux_data_size, obj_size, reader.size());
  // prev_reader_size = reader.size();
  // reader | unpacked_message->sent_across_nodes;
  // obj_size = reader.size() - prev_reader_size;
  // Parallel::printf(
  //     "unpack: unpacked bool: %s, bool size: %d, reader size: %d\n",
  //     unpacked_message->sent_across_nodes ? "true" : "false", obj_size,
  //     reader.size());
  // prev_reader_size = reader.size();
  // reader | unpacked_message->current_time_step_id;
  // obj_size = reader.size() - prev_reader_size;
  // Parallel::printf(
  //     "unpack: unpacked TimeStepId: %s, TimeStepId size: %d, reader
  //     size:%d\n", unpacked_message->current_time_step_id, obj_size,
  //     reader.size());
  // prev_reader_size = reader.size();
  // reader | unpacked_message->next_time_step_id;
  // obj_size = reader.size() - prev_reader_size;
  // Parallel::printf(
  //     "unpack: unpacked TimeStepId: %s, TimeStepId size: %d, reader
  //     size:%d\n", unpacked_message->next_time_step_id, obj_size,
  //     reader.size());
  // prev_reader_size = reader.size();
  // reader | unpacked_message->volume_or_ghost_mesh;
  // obj_size = reader.size() - prev_reader_size;
  // Parallel::printf(
  //     "unpack: unpacked Mesh<%d>: %s, Mesh<%d> size: %d, reader size: %d\n",
  //     Dim, unpacked_message->volume_or_ghost_mesh, Dim, obj_size,
  //     reader.size());
  // prev_reader_size = reader.size();
  // if constexpr (Dim > 1) {
  //   reader | unpacked_message->interface_mesh;
  // } else {
  //   unpacked_message->interface_mesh = Mesh<0>{};
  // }
  // obj_size = reader.size() - prev_reader_size;
  // Parallel::printf(
  //     "unpack: unpacked Mesh<%d>: %s, Mesh<%d> size: %d, reader size: %d\n",
  //     Dim, unpacked_message->interface_mesh, Dim, obj_size, reader.size());
  // prev_reader_size = reader.size();
  // Parallel::printf("unpack: sizeof unpacked_message: %d\n",
  //                  sizeof(*unpacked_message));
  // // If we actually have data, set the pointer, then call PUParray to copy
  // // the data
  // // std::stringstream ss{};
  // if (subcell_size != 0) {
  //   unpacked_message->subcell_ghost_data = reinterpret_cast<double*>(
  //       reinterpret_cast<char*>(unpacked_message) +
  //       total_size_without_data());
  //   PUParray(reader, unpacked_message->subcell_ghost_data, subcell_size);
  //   // ss << "(" << unpacked_message->subcell_ghost_data[0];
  //   // for (size_t i=1; i<subcell_size; i++) {
  //   //   ss << "," << unpacked_message->subcell_ghost_data[i];
  //   // }
  //   // ss << ")";
  //   (void)unpacked_message->subcell_ghost_data;
  // } else {
  //   // Otherwise just set the data to a nullptr
  //   unpacked_message->subcell_ghost_data = nullptr;
  //   // ss << "()";
  // }
  // Parallel::printf("unpack: sizeof unpacked_message: %d\n",
  //                  sizeof(*unpacked_message));
  // obj_size = reader.size() - prev_reader_size;
  // Parallel::printf("unpack: subcell data size: %d, reader size: %d\n",
  // obj_size,
  //                  reader.size());
  // prev_reader_size = reader.size();
  // // Parallel::printf("unpack: unpacked subcell data = %s\n", ss.str());
  // // ss.str("");
  // if (dg_size != 0) {
  //   unpacked_message->dg_flux_data = reinterpret_cast<double*>(
  //       reinterpret_cast<char*>(unpacked_message) + total_size_without_data()
  //       + subcell_size * sizeof(double));
  //   PUParray(reader, unpacked_message->dg_flux_data, dg_size);
  //   // ss << "(" << unpacked_message->dg_flux_data[0];
  //   // for (size_t i=1; i<dg_size; i++) {
  //   //   ss << "," << unpacked_message->dg_flux_data[i];
  //   // }
  //   // ss << ")";
  // } else {
  //   unpacked_message->dg_flux_data = nullptr;
  //   // ss << "()";
  // }
  // Parallel::printf("unpack: sizeof unpacked_message: %d\n",
  //                  sizeof(*unpacked_message));
  // const size_t sizeof_boundary_message_components =
  //     2 * sizeof(size_t) + sizeof(bool) + 2 * sizeof(TimeStepId) +
  //     sizeof(Mesh<Dim>) + sizeof(Mesh<Dim - 1>) + 2 * sizeof(double*);
  // Parallel::printf("unpack: sizof boundary message components: %d\n",
  //                  sizeof_boundary_message_components);
  // obj_size = reader.size() - prev_reader_size;
  // Parallel::printf("unpack: dg data size: %d, reader size: %d\n", obj_size,
  //                  reader.size());
  // Parallel::printf("unpack: unpacked dg data = %s\n", ss.str());

  // DEBUG: memcpy approach....not working
  (void)print;
  // char* buffer = reinterpret_cast<char*>(inbuf);

  BoundaryMessage<Dim>* buffer = reinterpret_cast<BoundaryMessage<Dim>*>(inbuf);

  const size_t subcell_size = buffer->subcell_ghost_data_size;
  const size_t dg_size = buffer->dg_flux_data_size;

  if (subcell_size != 0) {
    buffer->subcell_ghost_data =
        reinterpret_cast<double*>(std::addressof(buffer->dg_flux_data)) +
        detail::offset<double*>();
  } else {
    buffer->subcell_ghost_data = nullptr;
  }
  if (dg_size != 0) {
    buffer->dg_flux_data =
        reinterpret_cast<double*>(std::addressof(buffer->dg_flux_data)) +
        detail::offset<double*>() + subcell_size;
  } else {
    buffer->dg_flux_data = nullptr;
  }

  // Parallel::printf("Things %s, %s, %s, %s\n",
  // temp_buffer->current_time_step_id,
  //                  temp_buffer->next_time_step_id,
  //                  temp_buffer->volume_or_ghost_mesh,
  //                  temp_buffer->interface_mesh);

  // const size_t subcell_size = temp_buffer->subcell_ghost_data_size;
  // const size_t dg_size = temp_buffer->dg_flux_data_size;

  // const size_t totalsize =
  //     total_size_with_data_and_pointers(subcell_size, dg_size);

  // BoundaryMessage<Dim>* unpacked_message =
  //     reinterpret_cast<BoundaryMessage<Dim>*>(CkAllocBuffer(buffer,
  //     totalsize));

  // std::cout << "Address of unpacked message = "
  //           << std::addressof(unpacked_message);
  // std::cout << "\nunpacked message ptr = " << unpacked_message << "\n";

  // memcpy(&unpacked_message->subcell_ghost_data_size, buffer,
  //        detail::offset<size_t>());
  // buffer += detail::offset<size_t>();
  // memcpy(&unpacked_message->dg_flux_data_size, buffer,
  //        detail::offset<size_t>());
  // buffer += detail::offset<size_t>();
  // memcpy(&unpacked_message->sent_across_nodes, buffer,
  // detail::offset<bool>()); buffer += detail::offset<bool>();
  // memcpy(&unpacked_message->current_time_step_id, buffer,
  //        detail::offset<TimeStepId>());
  // buffer += detail::offset<TimeStepId>();
  // memcpy(&unpacked_message->next_time_step_id, buffer,
  //        detail::offset<TimeStepId>());
  // buffer += detail::offset<TimeStepId>();
  // memcpy(&unpacked_message->volume_or_ghost_mesh, buffer,
  //        detail::offset<Mesh<Dim>>());
  // buffer += detail::offset<Mesh<Dim>>();
  // memcpy(&unpacked_message->interface_mesh, buffer,
  //        detail::offset<Mesh<Dim - 1>>());
  // buffer += detail::offset<Mesh<Dim - 1>>();
  // std::cout << "Sizes: " << unpacked_message->subcell_ghost_data_size << " "
  //           << unpacked_message->dg_flux_data_size << "\n";
  // Parallel::printf(
  //     "Things %s, %s, %s, %s\n", unpacked_message->current_time_step_id,
  //     unpacked_message->next_time_step_id,
  //     unpacked_message->volume_or_ghost_mesh,
  //     unpacked_message->interface_mesh);
  // std::cout << "Addresses:\n"
  //           << std::addressof(unpacked_message->volume_or_ghost_mesh) << "\n"
  //           << std::addressof(unpacked_message->interface_mesh) << "\n"
  //           << std::addressof(unpacked_message->subcell_ghost_data) << "\n"
  //           << std::addressof(unpacked_message->dg_flux_data) << "\n";
  // // We don't need to set either of the pointers before we do a memcpy
  // // because
  // // only one will be a valid pointer at a time so whichever one it is will
  // // be
  // // the address right after Mesh<Dim>. The other will be a nullptr
  // if (subcell_size != 0) {
  //   unpacked_message->dg_flux_data = nullptr;
  //   // unpacked_message->subcell_ghost_data =
  //   //     reinterpret_cast<double*>(reinterpret_cast<char*>(std::addressof(
  //   //                                   unpacked_message->dg_flux_data)) +
  //   //                               8);
  //   std::cout << unpacked_message->subcell_ghost_data << "\n";
  //   std::cout << unpacked_message->subcell_ghost_data[0] << ","
  //             << unpacked_message->subcell_ghost_data[1] << ","
  //             << unpacked_message->subcell_ghost_data[2] << "\n";

  //   memcpy(unpacked_message->subcell_ghost_data, buffer, subcell_size);
  //   std::cout << unpacked_message->subcell_ghost_data[0] << ","
  //             << unpacked_message->subcell_ghost_data[1] << ","
  //             << unpacked_message->subcell_ghost_data[2] << "\n";

  //   buffer += subcell_size;
  // } else if (dg_size != 0) {
  //   unpacked_message->subcell_ghost_data = nullptr;
  //   memcpy(unpacked_message->dg_flux_data, buffer, dg_size);
  // }

  // Gotta clean up
  // CkFreeMsg(inbuf);
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
