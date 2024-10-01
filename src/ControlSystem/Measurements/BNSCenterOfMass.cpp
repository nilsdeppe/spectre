// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/Measurements/BNSCenterOfMass.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace control_system::measurements {

void center_of_mass_integral_on_element(
    const gsl::not_null<double*> mass_a, const gsl::not_null<double*> mass_b,
    const gsl::not_null<std::array<double, 3>*> first_moment_a,
    const gsl::not_null<std::array<double, 3>*> first_moment_b,
    const Mesh<3>& mesh, const Scalar<DataVector>& inv_det_jacobian,
    const Scalar<DataVector>& tilde_d,
    const tnsr::I<DataVector, 3, Frame::Grid>& x_grid) {
  // Get Jacobian and mask for left/right stars
  const DataVector det_jacobian = 1. / get(inv_det_jacobian);
  const DataVector positive_x = step_function(get<0>(x_grid));
  const DataVector negative_x = (1.0 - positive_x);

  // Integrals of the density and its first moment (on local element).
  // Suffix A/B for positive/negative x-coordinate (proxy for stars A and B)
  const DataVector integrand_a = det_jacobian * positive_x * get(tilde_d);
  const DataVector integrand_b = det_jacobian * negative_x * get(tilde_d);

  (*mass_a) = definite_integral(integrand_a, mesh);
  (*mass_b) = definite_integral(integrand_b, mesh);
  for (size_t i = 0; i < 3; i++) {
    gsl::at(*first_moment_a, i) =
        definite_integral(integrand_a * x_grid.get(i), mesh);
    gsl::at(*first_moment_b, i) =
        definite_integral(integrand_b * x_grid.get(i), mesh);
  }
}
}  // namespace control_system::measurements
