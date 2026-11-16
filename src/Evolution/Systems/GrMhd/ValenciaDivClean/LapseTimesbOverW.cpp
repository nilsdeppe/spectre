// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/LapseTimesbOverW.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean {
void magnetic_field_dot_spatial_velocity_and_lapse_times_b_over_w(
    const gsl::not_null<Scalar<DataVector>*>
        magnetic_field_dot_spatial_velocity,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        lapse_b_over_w,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3, Frame::Inertial>& spatial_velocity_one_form,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  dot_product(magnetic_field_dot_spatial_velocity, magnetic_field,
              spatial_velocity_one_form);
  for (size_t i = 0; i < 3; ++i) {
    // Start with the magnetic field one-form B_i
    lapse_b_over_w->get(i) = magnetic_field.get(0) * spatial_metric.get(i, 0);
    for (size_t j = 1; j < 3; ++j) {
      lapse_b_over_w->get(i) +=
          magnetic_field.get(j) * spatial_metric.get(i, j);
    }
    lapse_b_over_w->get(i) =
        get(lapse) * (lapse_b_over_w->get(i) / square(get(lorentz_factor)) +
                      spatial_velocity_one_form.get(i) *
                          get(*magnetic_field_dot_spatial_velocity));
  }
}
}  // namespace grmhd::ValenciaDivClean
