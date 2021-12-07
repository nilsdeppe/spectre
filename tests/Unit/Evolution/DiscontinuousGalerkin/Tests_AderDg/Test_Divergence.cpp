// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/AderDg/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace AderDg {

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t VolumeDim>
auto make_affine_map();

template <>
auto make_affine_map<1>() {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine{-1.0, 1.0, -0.3, 0.7});
}

template <>
auto make_affine_map<2>() {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine2D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
}

template <>
auto make_affine_map<3>() {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine3D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
               Affine{-1.0, 1.0, 2.3, 2.8}});
}

template <size_t Dim, typename Frame = Frame::Inertial>
void test_divergence_time_independent_jacobian(
    const Mesh<Dim>& mesh,
    std::array<std::unique_ptr<MathFunction<1>>, Dim> functions) {
  const auto coordinate_map = make_affine_map<Dim>();
  const size_t num_grid_points = mesh.number_of_grid_points();
  const auto xi = logical_coordinates(mesh);
  const auto x = coordinate_map(xi);
  const auto inv_jacobian = coordinate_map.inv_jacobian(xi);

  MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.AderDg.Divergence", "[Unit][Ader]") {
  const size_t n0 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2;
  const size_t n1 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2 + 1;
  const size_t n2 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2 - 1;
  const Mesh<1> mesh_1d{
      {{n0}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const Mesh<2> mesh_2d{{{n0, n1}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> mesh_3d{{{n0, n1, n2}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const size_t number_temporal_pts =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2 - 1;

  for (size_t a = 0; a < 5; ++a) {
    std::array<std::unique_ptr<MathFunction<1>>, 1> functions_1d{
        {std::make_unique<MathFunctions::PowX>(a)}};
    test_divergence_time_independent_jacobian(mesh_1d, std::move(functions_1d));
    for (size_t b = 0; b < 4; ++b) {
      std::array<std::unique_ptr<MathFunction<1>>, 2> functions_2d{
          {std::make_unique<MathFunctions::PowX>(a),
           std::make_unique<MathFunctions::PowX>(b)}};
      test_divergence_time_independent_jacobian(mesh_2d,
                                                std::move(functions_2d));
      for (size_t c = 0; c < 3; ++c) {
        std::array<std::unique_ptr<MathFunction<1>>, 3> functions_3d{
            {std::make_unique<MathFunctions::PowX>(a),
             std::make_unique<MathFunctions::PowX>(b),
             std::make_unique<MathFunctions::PowX>(c)}};
        test_divergence_time_independent_jacobian(mesh_3d,
                                                  std::move(functions_3d));
      }
    }
  }
}
}  // namespace AderDg
