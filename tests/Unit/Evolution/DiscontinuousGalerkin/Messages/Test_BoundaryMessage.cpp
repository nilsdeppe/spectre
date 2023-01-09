// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include <iostream>
#include "Parallel/Printf.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim, typename Generator>
void test_boundary_message(const gsl::not_null<Generator*> generator,
                           const size_t subcell_size, const size_t dg_size) {
  CAPTURE(Dim);
  CAPTURE(subcell_size);
  CAPTURE(dg_size);
  Parallel::printf("\ntest: subcell size = %d\n", subcell_size);
  Parallel::printf("test: dg size = %d\n", dg_size);
  const bool sent_across_nodes = true;

  const Slab current_slab{0.1, 0.5};
  const Time current_time{current_slab, {0, 1}};
  const TimeStepId current_time_id{true, 0, current_time};
  const Slab next_slab{0.5, 0.9};
  const Time next_time{next_slab, {0, 1}};
  const TimeStepId next_time_id{true, 0, next_time};

  const size_t extents = 4;
  const Mesh<Dim> volume_mesh{extents, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim - 1> interface_mesh = volume_mesh.slice_away(0);

  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  auto subcell_data = make_with_random_values<DataVector>(
      generator, make_not_null(&dist), subcell_size);
  auto dg_data = make_with_random_values<DataVector>(
      generator, make_not_null(&dist), dg_size);

  BoundaryMessage<Dim>* boundary_message = new BoundaryMessage<Dim>(
      subcell_size, dg_size, sent_across_nodes, current_time_id, next_time_id,
      volume_mesh, interface_mesh,
      subcell_size != 0 ? subcell_data.data() : nullptr,
      dg_size != 0 ? dg_data.data() : nullptr);

  CHECK(subcell_data.size() == subcell_size);
  CHECK(dg_data.size() == dg_size);

  Parallel::printf("sizeof(BoundaryMessage<%d>) = %d\n", Dim,
                   sizeof(BoundaryMessage<Dim>));
  Parallel::printf("alignof(BoundaryMessage<%d>) = %d\n", Dim,
                   alignof(BoundaryMessage<Dim>));
  Parallel::printf("sizeof(CMessage_BoundaryMessage<%d>) = %d\n", Dim,
                   sizeof(CMessage_BoundaryMessage<Dim>));
  Parallel::printf("sizeof(CkMessage) = %d\n", sizeof(CkMessage));
  Parallel::printf("sizeof(size_t) = %d, alignof(size_t) = %d\n",
                   sizeof(size_t), alignof(size_t));
  Parallel::printf("sizeof(bool) = %d, alignof(bool) = %d\n", sizeof(bool),
                   alignof(bool));
  Parallel::printf("sizeof(TimeStepId) = %d, alignof(TimeStepId) = %d\n",
                   sizeof(TimeStepId), alignof(TimeStepId));
  Parallel::printf("sizeof(Mesh<%d>) = %d, alignof(Mesh<%d>) = %d\n", Dim,
                   sizeof(Mesh<Dim>), Dim, alignof(Mesh<Dim>));
  Parallel::printf("sizeof(Mesh<%d>) = %d, alignof(Mesh<%d>) = %d\n", Dim - 1,
                   sizeof(Mesh<Dim - 1>), Dim - 1, alignof(Mesh<Dim - 1>));
  Parallel::printf("sizeof(double*) = %d, alignof(double*) = %d\n",
                   sizeof(double*), alignof(double*));
  size_t size = 2 * sizeof(size_t) + sizeof(bool) + 2 * sizeof(TimeStepId) +
                sizeof(Mesh<Dim>) + sizeof(Mesh<Dim - 1>);
  Parallel::printf("2 * sizeof(size_t) + ... (no pointers) = %d\n", size);
  size += 2 * sizeof(double*);
  Parallel::printf("2 * sizeof(size_t) + ... (yes pointers) = %d\n", size);
  size = 2 * sizeof(size_t) + 8 + 2 * sizeof(TimeStepId) + sizeof(Mesh<Dim>) +
         sizeof(Mesh<Dim - 1>);
  Parallel::printf("2 * sizeof(size_t) + ... (bool = 8, no pointers) = %d\n",
                   size);
  size += 2 * sizeof(double*);
  Parallel::printf("2 * sizeof(size_t) + ... (bool = 8, yes pointers) = %d\n",
                   size);
  PUP::sizer sizer;
  sizer | boundary_message->subcell_ghost_data_size;
  sizer | boundary_message->dg_flux_data_size;
  sizer | boundary_message->sent_across_nodes;
  sizer | boundary_message->current_time_step_id;
  sizer | boundary_message->next_time_step_id;
  sizer | boundary_message->volume_or_ghost_mesh;
  sizer | boundary_message->interface_mesh;
  Parallel::printf("sizer size (no pointers) = %d\n", sizer.size());

  //   BoundaryMessage<Dim>* copied_boundary_message =
  //       new (CkCopyMsg(reinterpret_cast<void**>(&boundary_message)))
  //           BoundaryMessage<Dim>();

  //   void* packed_message = boundary_message->pack(boundary_message);
  //   (void)packed_message;

  BoundaryMessage<Dim>* packed_and_unpacked_message =
      boundary_message->unpack(boundary_message->pack(boundary_message));
  Parallel::printf("After pack/unpack\n");

  std::cout << std::addressof(packed_and_unpacked_message->subcell_ghost_data)
            << "\n";
  std::cout << packed_and_unpacked_message->subcell_ghost_data << "\n";
  std::cout << std::addressof(packed_and_unpacked_message->dg_flux_data)
            << "\n";
  std::cout << packed_and_unpacked_message->dg_flux_data << "\n";

  Parallel::printf("\npacked/unpacked message = %s\n",
                   *packed_and_unpacked_message);
  //   Parallel::printf("boundary message = %s\n", *boundary_message);

  //   CHECK(*boundary_message == *packed_and_unpacked_message);

  // NOTE: Can't delete the pointers here????
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.BoundaryMessage", "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  std::uniform_int_distribution<size_t> size_dist{2, 10};
  tmpl::for_each<tmpl::integral_list<size_t, 0, 1, 2, 3>>([](auto dim_t) {
    constexpr size_t Dim =
        tmpl::type_from<std::decay_t<decltype(dim_t)>>::value;
    Parallel::printf("size of Mesh<%d> = %d\n", Dim, sizeof(Mesh<Dim>));
    Mesh<Dim> mesh{};
    Parallel::printf("size of Mesh<%d> in bytes = %d\n", Dim,
                     size_of_object_in_bytes(mesh));
  });

  (void)size_dist;
  test_boundary_message<1>(make_not_null(&generator), 3, 0);

  //   tmpl::for_each<tmpl::integral_list<size_t, /*1*/, 2, 3>>(
  //       [&generator, &size_dist](auto dim_t) {
  //         constexpr size_t Dim =
  //             tmpl::type_from<std::decay_t<decltype(dim_t)>>::value;
  //         test_boundary_message<Dim>(make_not_null(&generator),
  //                                    size_dist(generator), 0);
  //         test_boundary_message<Dim>(make_not_null(&generator), 0,
  //                                    size_dist(generator));
  //         // Although our use case is when *either* dg or subcell is zero, we
  //         // still test if both are non-zero just to check we can pack/unpack
  //         // correctly
  //         // test_boundary_message<Dim>(make_not_null(&generator),
  //         // size_dist(generator), size_dist(generator));
  //       });
}
}  // namespace
}  // namespace evolution::dg
