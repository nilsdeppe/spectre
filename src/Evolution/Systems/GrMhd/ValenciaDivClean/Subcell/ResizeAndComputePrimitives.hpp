// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief Mutator that resizes the primitive variables to have the size of the
 * active mesh and then computes the primitive variables on the active mesh.
 *
 * In the DG-subcell `step_actions` list this will normally be called using the
 * `::Actions::MutateApply` action right after the
 * `Actions::MutateApply<grmhd::ValenciaDivClean::subcell::SwapGrTags>` action.
 *
 * This mutator computes the primitive variables on the active grid. If the
 * active grid is DG that means we are switching from subcell to DG and the
 * primitive variables must still be the size of the subcell mesh. In this case
 * we reconstruct the pressure to the DG grid to give a high-order initial guess
 * for the primitive recovery. It would be nice if we can avoid this
 * reconstruction when all recovery schemes don't need an initial guess.
 * Finally, we perform the primitive recovery on the active grid.
 */
template <typename OrderedListOfRecoverySchemes>
struct ResizeAndComputePrims {
  using return_tags =
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>>;
  using argument_tags =
      tmpl::list<evolution::dg::subcell::Tags::ActiveGrid,
                 domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
                 grmhd::ValenciaDivClean::Tags::TildeD,
                 grmhd::ValenciaDivClean::Tags::TildeTau,
                 grmhd::ValenciaDivClean::Tags::TildeS<>,
                 grmhd::ValenciaDivClean::Tags::TildeB<>,
                 grmhd::ValenciaDivClean::Tags::TildePhi,
                 gr::Tags::SpatialMetric<3>, gr::Tags::InverseSpatialMetric<3>,
                 gr::Tags::SqrtDetSpatialMetric<>,
                 hydro::Tags::EquationOfStateBase>;

  template <size_t ThermodynamicDim>
  static void apply(
      gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*> prim_vars,
      evolution::dg::subcell::ActiveGrid active_grid, const Mesh<3>& dg_mesh,
      const Mesh<3>& subcell_mesh, const Scalar<DataVector>& tilde_d,
      const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          eos) noexcept;
};
}  // namespace grmhd::ValenciaDivClean::subcell
