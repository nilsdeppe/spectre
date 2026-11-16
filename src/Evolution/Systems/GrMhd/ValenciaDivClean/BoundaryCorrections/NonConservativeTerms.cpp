// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/NonConservativeTerms.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
void package_non_conservative_data(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_lapse_times_spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_normal_dot_lapse_times_spatial_velocity,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_covector,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_lapse_times_magnetic_field_dot_spatial_velocity,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_lapse_b_over_w,
    const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lapse_b_over_w,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector) {
  for (size_t i = 0; i < 3; ++i) {
    packaged_lapse_times_spatial_velocity->get(i) =
        get(lapse) * spatial_velocity.get(i);
  }
  dot_product(packaged_normal_dot_lapse_times_spatial_velocity,
              *packaged_lapse_times_spatial_velocity, normal_covector);
  *packaged_normal_covector = normal_covector;
  get(*packaged_lapse_times_magnetic_field_dot_spatial_velocity) =
      get(lapse) * get(magnetic_field_dot_spatial_velocity);
  *packaged_lapse_b_over_w = lapse_b_over_w;
}

namespace {
struct TildeBDotNormalInt : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct TildeBDotNormalExt : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct JumpTildeBDotNormalInt : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct JumpTildeBDotNormalExt : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

void add_non_conservative_boundary_terms(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_phi_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        lapse_times_spatial_velocity_int,
    const Scalar<DataVector>& normal_dot_lapse_times_spatial_velocity_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector_int,
    const Scalar<DataVector>&
        lapse_times_magnetic_field_dot_spatial_velocity_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lapse_b_over_w_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataVector>& tilde_phi_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        lapse_times_spatial_velocity_ext,
    const Scalar<DataVector>& normal_dot_lapse_times_spatial_velocity_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector_ext,
    const Scalar<DataVector>&
        lapse_times_magnetic_field_dot_spatial_velocity_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lapse_b_over_w_ext) {
  Variables<tmpl::list<TildeBDotNormalInt, TildeBDotNormalExt,
                       JumpTildeBDotNormalInt, JumpTildeBDotNormalExt>>
      buffer{get(tilde_phi_int).size()};
  // \tilde{B}^j n_j on each side, using that side's normal covector
  DataVector& tilde_b_dot_normal_int = get(get<TildeBDotNormalInt>(buffer));
  DataVector& tilde_b_dot_normal_ext = get(get<TildeBDotNormalExt>(buffer));
  dot_product(make_not_null(&get<TildeBDotNormalInt>(buffer)), tilde_b_int,
              normal_covector_int);
  dot_product(make_not_null(&get<TildeBDotNormalExt>(buffer)), tilde_b_ext,
              normal_covector_ext);
  // (\tilde{B}^j_ext - \tilde{B}^j_int) n_j with the normal covector of each
  // side
  DataVector& jump_tilde_b_dot_normal_int =
      get(get<JumpTildeBDotNormalInt>(buffer));
  DataVector& jump_tilde_b_dot_normal_ext =
      get(get<JumpTildeBDotNormalExt>(buffer));
  jump_tilde_b_dot_normal_int = 0.0;
  jump_tilde_b_dot_normal_ext = 0.0;
  for (size_t j = 0; j < 3; ++j) {
    jump_tilde_b_dot_normal_int +=
        (tilde_b_ext.get(j) - tilde_b_int.get(j)) * normal_covector_int.get(j);
    jump_tilde_b_dot_normal_ext +=
        (tilde_b_ext.get(j) - tilde_b_int.get(j)) * normal_covector_ext.get(j);
  }

  get(*boundary_correction_tilde_phi) +=
      0.25 * (get(tilde_phi_ext) - get(tilde_phi_int)) *
      (get(normal_dot_lapse_times_spatial_velocity_int) +
       get(normal_dot_lapse_times_spatial_velocity_ext));

  get(*boundary_correction_tilde_tau) +=
      0.25 * (get(tilde_phi_ext) - get(tilde_phi_int)) *
          (tilde_b_dot_normal_int + tilde_b_dot_normal_ext) +
      0.25 * (tilde_b_dot_normal_int - tilde_b_dot_normal_ext) *
          (get(lapse_times_magnetic_field_dot_spatial_velocity_int) +
           get(lapse_times_magnetic_field_dot_spatial_velocity_ext));

  for (size_t i = 0; i < 3; ++i) {
    boundary_correction_tilde_b->get(i) +=
        0.25 *
        (jump_tilde_b_dot_normal_int * lapse_times_spatial_velocity_int.get(i) +
         jump_tilde_b_dot_normal_ext * lapse_times_spatial_velocity_ext.get(i));
    boundary_correction_tilde_s->get(i) +=
        0.25 * (jump_tilde_b_dot_normal_int * lapse_b_over_w_int.get(i) +
                jump_tilde_b_dot_normal_ext * lapse_b_over_w_ext.get(i));
  }
}
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
