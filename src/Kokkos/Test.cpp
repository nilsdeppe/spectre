// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <Kokkos_Core.hpp>

#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

void foo();

template <typename T>
struct Referenced;

template <typename... Ts>
struct Referenced<Kokkos::View<Ts...>> : Kokkos::View<Ts...> {};

template <typename T>
struct Indexed {};

template <typename... Ts, typename Symm, template <typename...> class IndexList,
          typename... Indices>
class Tensor<Indexed<Kokkos::View<Ts...>>, Symm, IndexList<Indices...>> {
 public:
  // The type of the sequence that holds the data
  using storage_type =
      Kokkos::View<Ts...>[Tensor_detail::Structure<Symm, Indices...>::size()];
  // The type of the stored grid point indices
  using indices_type = int[Kokkos::View<Ts...>::rank()];
  // The type that is stored by the Tensor
  using type = Kokkos::View<Ts...>;
  // Typelist of the symmetry of the Tensor
  //
  // \details
  // For a rank-3 tensor symmetric in the last two indices,
  // \f$T_{a(bc)}\f$, the ::Symmetry is `<2, 1, 1>`. For a non-symmetric rank-2
  // tensor the ::Symmetry is `<2, 1>`.
  using symmetry = Symm;
  // Typelist of the \ref SpacetimeIndex "TensorIndexType"'s that the
  // Tensor has
  using index_list = tmpl::list<Indices...>;
  // The number of indices that the Tensor has
  static constexpr size_t num_tensor_indices = sizeof...(Indices);
  // The Tensor_detail::Structure for the particular tensor index structure
  //
  // Each tensor index structure, e.g. \f$T_{ab}\f$, \f$T_a{}^b\f$ or
  // \f$T^{ab}\f$ has its own Tensor_detail::TensorStructure that holds
  // information about how the data is stored, what the multiplicity of the
  // stored indices are, the number of (independent) components, etc.
  using structure = Tensor_detail::Structure<Symm, Indices...>;

  KOKKOS_FUNCTION Tensor() = default;
  template <typename... Ints>
  KOKKOS_FUNCTION Tensor(const Tensor<Kokkos::View<Ts...>, Symm,
                                      IndexList<Indices...>>& tensor_view,
                         Ints... ints)
      : indices_{static_cast<int>(ints)...} {
    static_assert(sizeof...(Ints) == Kokkos::View<Ts...>::rank());
    for (int i = 0; i < static_cast<int>(structure::size()); ++i) {
      data_[i] = tensor_view[static_cast<size_t>(i)];
    }
  }

  // // @{
  // // Retrieve the index `N...` by computing the storage index at compile time
  // // clang-tidy: redundant declaration (bug in clang-tidy)
  // template <int... N, typename... LocalTs, typename... Args>
  // friend SPECTRE_ALWAYS_INLINE
  //     typename Tensor<Kokkos::View<LocalTs...>, Args...>::reference
  //     get(Tensor<Kokkos::View<LocalTs...>, Args...>& t) {
  //   static_assert(
  //       Tensor<Kokkos::View<LocalTs...>, Args...>::rank() == sizeof...(N),
  //       "the number of tensor indices specified must match the rank "
  //       "of the tensor");
  //   return gsl::at(
  //       t.data_,
  //       Tensor<Kokkos::View<LocalTs...>,
  //              Args...>::structure::template get_storage_index<N...>());
  // }
  // // clang-tidy: redundant declaration (bug in clang-tidy)
  // template <int... N, typename... LocalTs, typename... Args>
  // friend SPECTRE_ALWAYS_INLINE
  //     typename Tensor<Kokkos::View<LocalTs...>, Args...>::const_reference
  //     get(const Tensor<Kokkos::View<LocalTs...>, Args...>& t) {
  //   static_assert(
  //       Tensor<Kokkos::View<LocalTs...>, Args...>::rank() == sizeof...(N),
  //       "the number of tensor indices specified must match the rank "
  //       "of the tensor");
  //   return gsl::at(
  //       t.data_,
  //       Tensor<Kokkos::View<LocalTs...>,
  //              Args...>::structure::template get_storage_index<N...>());
  // }
  // // @}

  template <typename... N>
  KOKKOS_FUNCTION typename Kokkos::View<Ts...>::reference_type get(N... n) {
    static_assert(
        sizeof...(Indices) == sizeof...(N),
        "the number of tensor indices specified must match the rank of "
        "the tensor");
    if constexpr (Kokkos::View<Ts...>::rank() == 1) {
      return data_[structure::get_storage_index(n...)](indices_[0]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 2) {
      return data_[structure::get_storage_index(n...)](indices_[0],
                                                       indices_[1]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 3) {
      return data_[structure::get_storage_index(n...)](indices_[0], indices_[1],
                                                       indices_[2]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 4) {
      return data_[structure::get_storage_index(n...)](
          indices_[0], indices_[1], indices_[2], indices_[3]);
    }
  }

  template <typename... N>
  KOKKOS_FUNCTION typename Kokkos::View<Ts...>::reference_type get(
      N... n) const {
    static_assert(
        sizeof...(Indices) == sizeof...(N),
        "the number of tensor indices specified must match the rank of "
        "the tensor");
    if constexpr (Kokkos::View<Ts...>::rank() == 1) {
      return data_[structure::get_storage_index(n...)](indices_[0]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 2) {
      return data_[structure::get_storage_index(n...)](indices_[0],
                                                       indices_[1]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 3) {
      return data_[structure::get_storage_index(n...)](indices_[0], indices_[1],
                                                       indices_[2]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 4) {
      return data_[structure::get_storage_index(n...)](
          indices_[0], indices_[1], indices_[2], indices_[3]);
    }
  }

 private:
  storage_type data_;
  indices_type indices_{};
};

template <class ExecSpace>
struct SpaceInstance {
  static ExecSpace create() { return ExecSpace(); }
  static void destroy(ExecSpace&) {}
  static bool overlap() { return false; }
};

template <>
struct SpaceInstance<Kokkos::Cuda> {
  static Kokkos::Cuda create() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return Kokkos::Cuda(stream);
  }
  static void destroy(Kokkos::Cuda& space) {
    cudaStream_t stream = space.cuda_stream();
    cudaStreamDestroy(stream);
  }
  static bool overlap() {
    bool value = true;
    auto local_rank_str = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (local_rank_str) {
      value = (std::stoi(local_rank_str) == 0);
    }
    return value;
  }
};

struct Test {
  KOKKOS_FUNCTION
  void operator()(int x, int y, int z) const {
    Scalar<Indexed<Kokkos::View<double***, Kokkos::CudaSpace>>> temperature{
        T, x, y, z},
        dt_temperature{dT, x, y, z};
    tnsr::ab<Indexed<Kokkos::View<double***, Kokkos::CudaSpace>>, 3,
             Frame::Inertial>
        spacetime_metric{full_spacetime_metric, x, y, z},
        dt_spacetime_metric{full_dt_spacetime_metric, x, y, z};
    // tnsr::ab<Indexed<Kokkos::View<double***, Kokkos::CudaSpace>>, 3,
    //          Frame::Inertial>
    //     spacetime_metric{}, dt_spacetime_metric{};
    // T.get()(x, y, z) += dt * dT.get()(x, y, z);
    // for (int i = 0; i < 3; ++i) {
    //   for (int j=i; j < 3; ++j) {
    //     spacetime_metric.get(i, j)(x, y, z) +=
    //         dt * dt_spacetime_metric.get(i, j)(x, y, z);
    //   }
    // }
    temperature.get() += dt * dt_temperature.get();
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        spacetime_metric.get(i, j) += dt * dt_spacetime_metric.get(i, j);
      }
    }
  }

  Scalar<Kokkos::View<double***, Kokkos::CudaSpace>> T, dT;
  tnsr::ab<Kokkos::View<double***, Kokkos::CudaSpace>, 3, Frame::Inertial>
      full_spacetime_metric{}, full_dt_spacetime_metric{};
  double t{0.0};
  double dt{0.1};
};

void foo() {
  auto E_bulk = SpaceInstance<Kokkos::DefaultExecutionSpace>::create();
  Test test{};
  using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, int>;
  Kokkos::parallel_for(
      "ComputeInnerDT",
      Kokkos::Experimental::require(
          policy_t(E_bulk, {1, 1, 1}, {200, 200, 200}),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
      test);
}
