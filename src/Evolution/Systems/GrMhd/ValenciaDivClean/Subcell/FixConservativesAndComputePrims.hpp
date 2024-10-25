// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <typename TagsList>
class Variables;
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief Fix the conservative variables and compute the primitive variables.
 *
 * Sets `ValenciaDivClean::Tags::VariablesNeededFixing` to `true` if the
 * conservative variables needed fixing, otherwise sets the tag to `false`.
 */
template <typename OrderedListOfRecoverySchemes>
struct FixConservativesAndComputePrims {
  using return_tags = tmpl::list<ValenciaDivClean::Tags::VariablesNeededFixing,
                                 typename System::variables_tag,
                                 typename System::primitive_variables_tag>;
  using argument_tags = tmpl::list<
      domain::Tags::Coordinates<3, Frame::Grid>,
      ::Tags::VariableFixer<grmhd::ValenciaDivClean::FixConservatives>,
      hydro::Tags::GrmhdEquationOfState, gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      grmhd::ValenciaDivClean::Tags::PrimitiveFromConservativeOptions>;

  static void apply(
      gsl::not_null<bool*> needed_fixing,
      gsl::not_null<typename System::variables_tag::type*> conserved_vars_ptr,
      gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*>
          primitive_vars_ptr,
      const tnsr::I<DataVector, 3, Frame::Grid>& dg_grid_coords,
      const grmhd::ValenciaDivClean::FixConservatives& fix_conservatives,
      const EquationsOfState::EquationOfState<true, 3>& eos,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
          primitive_from_conservative_options);
};
}  // namespace grmhd::ValenciaDivClean::subcell
