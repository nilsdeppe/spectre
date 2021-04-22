// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderSchemeLts.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticData/Burgers/Sinusoid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Bump.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Linear.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"                // IWYU pragma: keep
#include "Time/Actions/ChangeSlabSize.hpp"             // IWYU pragma: keep
#include "Time/Actions/ChangeStepSize.hpp"             // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"      // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"           // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                    // IWYU pragma: keep
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/Factory.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/Actions/SubcellCommunication.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/DgSubcell/Actions/TciAndRollback.hpp"
#include "Evolution/DgSubcell/Actions/TciAndSwitchToDg.hpp"
#include "Evolution/DgSubcell/CombineVolumeAndGhostData.hpp"
#include "Evolution/DgSubcell/Events/ObserveFields.hpp"
#include "Evolution/DgSubcell/FaceLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"

// IWYU pragma: no_include <pup.h>

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

template <size_t Index, bool PrintFieldsInfo>
struct PrintThings {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Parallel::printf("Finished self-start on %s\n",
    //                  db::get<subcell::Tags::ActiveGrid>(box));
    Parallel::printf("Id(%ld): %s %s %1.3e\n", Index,
                     db::get<domain::Tags::Element<1>>(box).id(),
                     db::get<evolution::dg::subcell::Tags::ActiveGrid>(box),
                     db::get<::Tags::Time>(box));

    if (Index == 30) {
      // using variables_tag = typename Metavariables::system::variables_tag;
      // using dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>;
      // using history_tag =
      //     Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>;
      // std::stringstream ss;
      // ss << "Id " << db::get<Tags::Element<1>>(box).id() << " History:\n";
      // for (auto hist_it = db::get<history_tag>(box).begin();
      //      hist_it != db::get<history_tag>(box).end(); ++hist_it) {
      //   ss << *hist_it << " " << hist_it.value() << "\n";
      // }
      // Parallel::printf("%s\n", ss.str());
    }
    return {std::move(box)};
  }
};

template <typename InitialData>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = 1;
  using system = Burgers::System;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;

  using initial_data = InitialData;
  static_assert(
      evolution::is_analytic_data_v<initial_data> xor
          evolution::is_analytic_solution_v<initial_data>,
      "initial_data must be either an analytic_data or an analytic_solution");
  using initial_data_tag =
      tmpl::conditional_t<evolution::is_analytic_solution_v<initial_data>,
                          Tags::AnalyticSolution<initial_data>,
                          Tags::AnalyticData<initial_data>>;

  // using limiter =
  //     Tags::Limiter<Limiters::Minmod<1, system::variables_tag::tags_list>>;

  using time_stepper_tag = Tags::TimeStepper<
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>>;

  using observe_fields = typename system::variables_tag::tags_list;
  using analytic_solution_fields =
      tmpl::conditional_t<evolution::is_analytic_solution_v<initial_data>,
                          observe_fields, tmpl::list<>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                Events::Completion,
                evolution::dg::subcell::Events::ObserveFields<
                    1, Tags::Time,
                    tmpl::push_back<observe_fields,
                                    evolution::dg::subcell::Tags::TciStatus>,
                    analytic_solution_fields>,
                Events::time_events<EvolutionMetavars>>>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   StepChoosers::standard_step_choosers<system>>,
        tmpl::pair<
            StepChooser<StepChooserUse::Slab>,
            StepChoosers::standard_slab_choosers<system, local_time_stepping>>,
        tmpl::pair<StepController, StepControllers::standard_step_controllers>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>>;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>>>>;
  struct SubcellOptions {
    static constexpr bool subcell_enabled = true;
    // We send `ghost_zone_size` cell-centered grid points for variable
    // reconstruction, of which we need `ghost_zone_size-1` for reconstruction
    // to the internal side of the element face, and `ghost_zone_size` for
    // reconstruction to the external side of the element face.
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& /*box*/) noexcept {
      return 2;
    };

    struct DgInitialDataTci {
      static bool apply(
          const Variables<tmpl::list<Burgers::Tags::U>>& dg_vars,
          const Variables<tmpl::list<
              evolution::dg::subcell::Tags::Inactive<Burgers::Tags::U>>>&
              subcell_vars,
          const Mesh<volume_dim>& dg_mesh, const double rdmp_delta0,
          const double rdmp_epsilon, const double persson_exponent) noexcept {
        const bool rdmp_troubled = evolution::dg::subcell::two_mesh_rdmp_tci(
            dg_vars, subcell_vars, rdmp_delta0, rdmp_epsilon);
        if (rdmp_troubled) {
          return rdmp_troubled;
        }
        return evolution::dg::subcell::persson_tci(
            get<Burgers::Tags::U>(dg_vars), dg_mesh, persson_exponent, 1.0e-18);
      }
    };

    template <bool UseInactiveGrid>
    struct TciOnDgGridImpl {
      using return_tags = tmpl::list<>;
      using argument_tags = tmpl::conditional_t<
          UseInactiveGrid,
          tmpl::list<evolution::dg::subcell::Tags::Inactive<Burgers::Tags::U>,
                     domain::Tags::Mesh<volume_dim>>,
          tmpl::list<Burgers::Tags::U>>;

      static bool apply(const Scalar<DataVector>& burgers_u,
                        const Mesh<volume_dim>& dg_mesh,
                        const double persson_exponent) noexcept {
        return evolution::dg::subcell::persson_tci(burgers_u, dg_mesh,
                                                   persson_exponent, 1.0e-18);
      }
    };

    using TciOnDgGrid = TciOnDgGridImpl<false>;
    using TciOnSubcellGrid = TciOnDgGridImpl<true>;

    struct TimeDerivative {
      template <typename DbTagsList>
      static void apply(
          const gsl::not_null<db::DataBox<DbTagsList>*> box,
          const Scalar<DataVector>& cell_centered_det_inv_jacobian) noexcept {
        const size_t ghost_zone_size = SubcellOptions::ghost_zone_size(*box);

        const size_t num_pts =
            db::get<evolution::dg::subcell::Tags::Mesh<volume_dim>>(*box)
                .number_of_grid_points();

        const auto face_log_coords =
            evolution::dg::subcell::fd::face_logical_coordinates(
                db::get<evolution::dg::subcell::Tags::Mesh<volume_dim>>(*box));
        const auto face_inv_jacobian =
            db::get<domain::Tags::ElementMap<volume_dim, Frame::Grid>>(*box)
                .inv_jacobian(face_log_coords);
        // We are solving using the logical formulation, and need the normal
        // vector to do so. Normalize on the faces, then split into upper/lower.
        using std::abs;
        const Scalar<DataVector> normal_magnitude{
            abs(get<0, 0>(face_inv_jacobian))};

        const Mesh<volume_dim>& subcell_mesh =
            db::get<evolution::dg::subcell::Tags::Mesh<volume_dim>>(*box);

        // Do reconstruction to upper and lower faces
        const auto volume_vars = db::get<typename system::variables_tag>(*box);
        Variables<tmpl::list<Burgers::Tags::U>> vars_with_ghosts{
            num_pts + 2 * ghost_zone_size};
        const auto& element = db::get<domain::Tags::Element<volume_dim>>(*box);
        ASSERT(
            element.external_boundaries().size() == 0,
            "Can't have external boundaries right now with subcell. ElementID "
                << element.id());
        const std::pair upper_mortar_id{
            Direction<volume_dim>::upper_xi(),
            *element.neighbors().at(Direction<volume_dim>::upper_xi()).begin()};
        const std::pair lower_mortar_id{
            Direction<volume_dim>::lower_xi(),
            *element.neighbors().at(Direction<volume_dim>::lower_xi()).begin()};
        const auto& nhbr_subcell_data =
            db::get<evolution::dg::subcell::Tags::
                        NeighborDataForReconstructionAndRdmpTci<volume_dim>>(
                *box);
        const auto& lower_ghost_data =
            nhbr_subcell_data.at(lower_mortar_id).data_for_reconstruction;
        const auto& upper_ghost_data =
            nhbr_subcell_data.at(upper_mortar_id).data_for_reconstruction;
        evolution::dg::subcell::combine_volume_and_ghost_data(
            make_not_null(&vars_with_ghosts), volume_vars, 0,
            subcell_mesh.extents(), ghost_zone_size,
            gsl::make_span(lower_ghost_data.data(), lower_ghost_data.size()),
            gsl::make_span(upper_ghost_data.data(), upper_ghost_data.size()));

        const Scalar<DataVector>& u_with_ghosts =
            get<Burgers::Tags::U>(vars_with_ghosts);

        // upper refers to the upper side of the interface
        Scalar<DataVector> u_face_upper{num_pts + 1};
        Scalar<DataVector> u_face_lower{num_pts + 1};

        const auto compute_sigma = [ghost_zone_size, &u_with_ghosts](
                                       const auto cell_index) noexcept {
          const auto gzs =
              static_cast<std::decay_t<decltype(cell_index)>>(ghost_zone_size);
          const double a =
              get(u_with_ghosts)[static_cast<size_t>(gzs + cell_index + 1)] -
              get(u_with_ghosts)[static_cast<size_t>(gzs + cell_index)];
          const double b =
              get(u_with_ghosts)[static_cast<size_t>(gzs + cell_index)] -
              get(u_with_ghosts)[static_cast<size_t>(gzs + cell_index - 1)];
          using std::min;
          using std::abs;
          // minmod
          // return 0.5 * (sgn(a) + sgn(b)) * min(abs(a), abs(b));
          // MC
          return 0.5 * (sgn(a) + sgn(b)) *
                 min(min(0.5 * abs(a + b), 2.0 * abs(a)), 2.0 * abs(b));
        };
        // Loop over cells since that's where we do reconstruction, but assign
        // to faces.
        for (size_t i = 0; i < num_pts; ++i) {
          const double sigma = compute_sigma(i);
          get(u_face_upper)[i] =
              get(u_with_ghosts)[ghost_zone_size + i] - 0.5 * sigma;
          get(u_face_lower)[i + 1] =
              get(u_with_ghosts)[ghost_zone_size + i] + 0.5 * sigma;
        }
        // Reconstruct to upper side of lower ghost cell, which is the lower
        // side of the face
        get(u_face_lower)[0] =
            get(u_with_ghosts)[ghost_zone_size - 1] + 0.5 * compute_sigma(-1);
        // Reconstruct to lower side of upper ghost cell, which is the upper
        // side of the face
        get(u_face_upper)[num_pts] =
            get(u_with_ghosts)[ghost_zone_size + num_pts] -
            0.5 * compute_sigma(num_pts);

        // Compute fluxes, then can compute boundary corrections.
        // TODO: why did num_pts without the + 1 ever work?
        tnsr::I<DataVector, 1, Frame::Inertial> flux_upper{num_pts + 1};
        Burgers::Fluxes::apply(make_not_null(&flux_upper), u_face_upper);
        tnsr::I<DataVector, 1, Frame::Inertial> flux_lower{num_pts + 1};
        Burgers::Fluxes::apply(make_not_null(&flux_lower), u_face_lower);

        // Normal vectors are easy, flat space. Note that we use the sign
        // convention on the normal vectors to be compatible with DG.
        const tnsr::i<DataVector, volume_dim, Frame::Inertial>
            upper_outward_conormal{num_pts + 1, -1.0};
        const tnsr::i<DataVector, volume_dim, Frame::Inertial>
            lower_outward_conormal{num_pts + 1, 1.0};

        // Now package the data and compute the correction
        const auto& boundary_correction =
            db::get<evolution::Tags::BoundaryCorrection<system>>(*box);
        using derived_boundary_corrections = typename std::decay_t<decltype(
            boundary_correction)>::creatable_classes;
        const auto& mortar_data =
            db::get<evolution::dg::Tags::MortarData<volume_dim>>(*box);
        tmpl::for_each<derived_boundary_corrections>(
            [&](auto derived_correction_v) noexcept {
              using DerivedCorrection =
                  tmpl::type_from<decltype(derived_correction_v)>;
              if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
                // Compute the packaged data
                using dg_package_field_tags =
                    typename DerivedCorrection::dg_package_field_tags;
                Variables<dg_package_field_tags> upper_packaged_data{num_pts +
                                                                     1};
                dynamic_cast<const DerivedCorrection&>(boundary_correction)
                    .dg_package_data(
                        make_not_null(
                            &get<Burgers::Tags::U>(upper_packaged_data)),
                        make_not_null(
                            &get<::Tags::NormalDotFlux<Burgers::Tags::U>>(
                                upper_packaged_data)),
                        make_not_null(&get<tmpl::back<dg_package_field_tags>>(
                            upper_packaged_data)),
                        u_face_upper, flux_upper, upper_outward_conormal,
                        std::nullopt, std::nullopt);
                Variables<dg_package_field_tags> lower_packaged_data{num_pts +
                                                                     1};
                dynamic_cast<const DerivedCorrection&>(boundary_correction)
                    .dg_package_data(
                        make_not_null(
                            &get<Burgers::Tags::U>(lower_packaged_data)),
                        make_not_null(
                            &get<::Tags::NormalDotFlux<Burgers::Tags::U>>(
                                lower_packaged_data)),
                        make_not_null(&get<tmpl::back<dg_package_field_tags>>(
                            lower_packaged_data)),
                        u_face_lower, flux_lower, lower_outward_conormal,
                        std::nullopt, std::nullopt);
                // Now need to check if any of our neighbors are doing DG,
                // because if so then we need to use whatever boundary data they
                // sent instead of what we computed locally. We should check
                // this beforehand to avoid the extra work.
                if (auto nhbr_mortar_data_it =
                        mortar_data.find(upper_mortar_id);
                    nhbr_mortar_data_it != mortar_data.end() and
                    nhbr_mortar_data_it->second.neighbor_mortar_data()
                        .has_value()) {
                  const size_t num_pts_on_dg_face = 1;
                  Variables<dg_package_field_tags> nhbr_packaged_data{
                      num_pts_on_dg_face};
                  const std::vector<double>& nhbr_data =
                      nhbr_mortar_data_it->second.neighbor_mortar_data()
                          ->second;
                  ASSERT(nhbr_packaged_data.size() == nhbr_data.size(),
                         "Trying to copy upper neighbor's packaged data from "
                         "vector of size "
                             << nhbr_data.size() << " into a Variables of size "
                             << nhbr_packaged_data.size());
                  std::copy(nhbr_data.begin(), nhbr_data.end(),
                            nhbr_packaged_data.data());
                  tmpl::for_each<dg_package_field_tags>(
                      [&nhbr_packaged_data, &num_pts,
                       &upper_packaged_data](auto tag_v) noexcept {
                        using tag = tmpl::type_from<decltype(tag_v)>;
                        ASSERT(get<tag>(nhbr_packaged_data).size() == 1,
                               "In 1d should only have 1 point on the face");
                        for (size_t storage_index = 0;
                             storage_index <
                             get<tag>(upper_packaged_data).size();
                             ++storage_index) {
                          get<tag>(
                              upper_packaged_data)[storage_index][num_pts] =
                              get<tag>(nhbr_packaged_data)[storage_index][0];
                        }
                      });
                }
                if (auto nhbr_mortar_data_it =
                        mortar_data.find(lower_mortar_id);
                    nhbr_mortar_data_it != mortar_data.end() and
                    nhbr_mortar_data_it->second.neighbor_mortar_data()
                        .has_value()) {
                  const size_t num_pts_on_dg_face = 1;
                  Variables<dg_package_field_tags> nhbr_packaged_data{
                      num_pts_on_dg_face};
                  const std::vector<double>& nhbr_data =
                      nhbr_mortar_data_it->second.neighbor_mortar_data()
                          ->second;
                  ASSERT(nhbr_packaged_data.size() == nhbr_data.size(),
                         "Trying to copy lower neighbor's packaged data from "
                         "vector of size "
                             << nhbr_data.size() << " into a Variables of size "
                             << nhbr_packaged_data.size());
                  std::copy(nhbr_data.begin(), nhbr_data.end(),
                            nhbr_packaged_data.data());
                  tmpl::for_each<dg_package_field_tags>(
                      [&nhbr_packaged_data,
                       &lower_packaged_data](auto tag_v) noexcept {
                        using tag = tmpl::type_from<decltype(tag_v)>;
                        ASSERT(get<tag>(nhbr_packaged_data).size() == 1,
                               "In 1d should only have 1 point on the face");
                        for (size_t storage_index = 0;
                             storage_index <
                             get<tag>(lower_packaged_data).size();
                             ++storage_index) {
                          get<tag>(lower_packaged_data)[storage_index][0] =
                              get<tag>(nhbr_packaged_data)[storage_index][0];
                        }
                      });
                }
                // TODO: use fluxes sent to neighbor if we did a rollback.

                // Compute the corrections on the faces. We only need to compute
                // this once because we can just flip the normal vectors then
                Scalar<DataVector> u_boundary_correction{num_pts + 1};
                using char_speed_tag = tmpl::back<dg_package_field_tags>;
                dynamic_cast<const DerivedCorrection&>(boundary_correction)
                    .dg_boundary_terms(
                        make_not_null(&u_boundary_correction),
                        get<Burgers::Tags::U>(upper_packaged_data),
                        get<::Tags::NormalDotFlux<Burgers::Tags::U>>(
                            upper_packaged_data),
                        get<char_speed_tag>(upper_packaged_data),

                        get<Burgers::Tags::U>(lower_packaged_data),
                        get<::Tags::NormalDotFlux<Burgers::Tags::U>>(
                            lower_packaged_data),
                        get<char_speed_tag>(lower_packaged_data),
                        // FD schemes are basically weak form FV scheme
                        dg::Formulation::WeakInertial);

                // Now compute the actual time derivatives.
                // We broke ass honkies, so we do 2nd order FD
                using variables_tag = typename system::variables_tag;
                using dt_variables_tag =
                    db::add_tag_prefix<::Tags::dt, variables_tag>;
                db::mutate<dt_variables_tag>(
                    box, [&cell_centered_det_inv_jacobian, &num_pts,
                          &u_boundary_correction,
                          delta_xi =
                              static_cast<double>(get<0>(face_log_coords)[1] -
                                                  get<0>(face_log_coords)[0])](
                             const auto dt_vars_ptr) noexcept {
                      if (dt_vars_ptr->number_of_grid_points() != num_pts) {
                        dt_vars_ptr->initialize(num_pts);
                      }
                      auto& dt_u =
                          get<Tags::dt<Burgers::Tags::U>>(*dt_vars_ptr);
                      for (size_t i = 0; i < num_pts; ++i) {
                        get(dt_u)[i] = (1.0 / delta_xi) *
                                       get(cell_centered_det_inv_jacobian)[i] *
                                       (get(u_boundary_correction)[i + 1] -
                                        get(u_boundary_correction)[i]);
                      }
                    });
              }
            });
      }
    };

    struct DgOuterCorrectionPackageData {
      template <typename DbTagsList>
      static FixedHashMap<
          maximum_number_of_neighbors(volume_dim),
          std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
          std::vector<double>,
          boost::hash<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>
      apply(const db::DataBox<DbTagsList>& box,
            const std::vector<
                std::pair<Direction<volume_dim>, ElementId<volume_dim>>>&
                mortars_to_reconstruct_to) noexcept {
        const size_t ghost_zone_size = SubcellOptions::ghost_zone_size(box);
        ASSERT(not db::get<domain::Tags::MeshVelocity<volume_dim>>(box)
                       .has_value(),
               "Haven't yet added support for moving mesh to DG-subcell. This "
               "should be easy to generalize, but we will want to consider "
               "storing the mesh velocity on the faces instead of "
               "re-slicing/projecting.");

        FixedHashMap<maximum_number_of_neighbors(volume_dim),
                     std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
                     std::vector<double>,
                     boost::hash<std::pair<Direction<volume_dim>,
                                           ElementId<volume_dim>>>>
            nhbr_package_data{};

        const auto& nhbr_subcell_data =
            db::get<evolution::dg::subcell::Tags::
                        NeighborDataForReconstructionAndRdmpTci<volume_dim>>(
                box);
        const Mesh<volume_dim>& subcell_mesh =
            db::get<evolution::dg::subcell::Tags::Mesh<volume_dim>>(box);
        // Since the active grid is DG, we need to use the inactive grid for
        // reconstruction.
        const auto& volume_u =
            db::get<evolution::dg::subcell::Tags::Inactive<Burgers::Tags::U>>(
                box);

        const auto& boundary_correction =
            db::get<evolution::Tags::BoundaryCorrection<system>>(box);
        using derived_boundary_corrections = typename std::decay_t<decltype(
            boundary_correction)>::creatable_classes;
        tmpl::for_each<derived_boundary_corrections>(
            [&box, &boundary_correction, &ghost_zone_size,
             &mortars_to_reconstruct_to, &nhbr_package_data, &nhbr_subcell_data,
             &subcell_mesh, &volume_u](auto derived_correction_v) noexcept {
              using DerivedCorrection =
                  tmpl::type_from<decltype(derived_correction_v)>;
              if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
                for (const auto& mortar_id : mortars_to_reconstruct_to) {
                  const evolution::dg::subcell::NeighborData& nhbr_data =
                      nhbr_subcell_data.at(mortar_id);
                  const Direction<1>& direction = mortar_id.first;
                  const size_t num_pts_face = 1;
                  Scalar<DataVector> u{num_pts_face};

                  // Just do MC reconstruction for now, we can do fancier later
                  // on.
                  ASSERT(nhbr_data.data_for_reconstruction.size() ==
                             ghost_zone_size,
                         "Should've received "
                             << ghost_zone_size
                             << " grid points for reconstruction, but got "
                             << nhbr_data.data_for_reconstruction.size());
                  ASSERT(ghost_zone_size == 2,
                         "Currently hard-coded to 2 for MC.");
                  const auto& nhbr_u = nhbr_data.data_for_reconstruction;
                  const bool upper_side = direction.side() == Side::Upper;
                  const size_t volume_index =
                      upper_side ? subcell_mesh.extents(0) - 1 : 0;
                  const double u_jm1 =
                      upper_side ? get(volume_u)[volume_index] : nhbr_u[0];
                  const double u_j = upper_side ? nhbr_u[0] : nhbr_u[1];
                  const double u_jp1 =
                      upper_side ? nhbr_u[1] : get(volume_u)[volume_index];
                  // a = 0.5 (u_{j+1} - u_{j-1})
                  const double a = 0.5 * (u_jp1 - u_jm1);
                  // b = 2.0 * (u_j - u_{j-1})
                  const double b = 2.0 * (u_j - u_jm1);
                  // b = 2.0 * (u_{j+1} - u_{j})
                  const double c = 2.0 * (u_jp1 - u_j);
                  // sigma is the undivided slope in the element
                  double sigma = 0.0;
                  if (a > 0.0 and b > 0.0 and c > 0.0) {
                    using std::min;
                    sigma = min(min(a, b), c);
                  } else if (a < 0.0 and b < 0.0 and c < 0.0) {
                    using std::max;
                    sigma = max(max(a, b), c);
                  }
                  // u_{j\pm1/2} = u_j \pm sigma / 2, on upper side of element
                  // we want lower edge, so extra minus sign
                  get(u)[0] = u_j - direction.sign() * 0.5 * sigma;

                  // Compute the flux
                  tnsr::I<DataVector, 1, Frame::Inertial> flux{num_pts_face};
                  Burgers::Fluxes::apply(make_not_null(&flux), u);

                  tnsr::i<DataVector, 1, Frame::Inertial> normal_covector =
                      get<evolution::dg::Tags::NormalCovector<volume_dim>>(
                          *db::get<evolution::dg::Tags::
                                       NormalCovectorAndMagnitude<volume_dim>>(
                               box)
                               .at(direction));
                  for (auto& t : normal_covector) {
                    t *= -1.0;
                  }

                  // Note: in 2d and 3d we'd need to project the normal vector
                  // to the subcells.

                  // Compute the packaged data
                  Variables<typename DerivedCorrection::dg_package_field_tags>
                      packaged_data{num_pts_face};
                  // TODO: we will need to figure out how to generalize all this
                  // crap...
                  // Note: moving mesh is disable, one step at a time...
                  dynamic_cast<const DerivedCorrection&>(boundary_correction)
                      .dg_package_data(
                          make_not_null(&get<Burgers::Tags::U>(packaged_data)),
                          make_not_null(
                              &get<::Tags::NormalDotFlux<Burgers::Tags::U>>(
                                  packaged_data)),
                          make_not_null(
                              &get<tmpl::back<typename DerivedCorrection::
                                                  dg_package_field_tags>>(
                                  packaged_data)),
                          u, flux, normal_covector, std::nullopt, std::nullopt);
                  nhbr_package_data[mortar_id] = std::vector<double>{
                      packaged_data.data(),
                      packaged_data.data() + packaged_data.size()};
                }
              }
            });
        return nhbr_package_data;
      }
    };

    struct PrepareGhostAndSubcellData {
      template <typename DbTagsList>
      static void apply(
          const gsl::not_null<db::DataBox<DbTagsList>*> /*box*/) noexcept {
        // For Burgers we send the evolved variables (there are no primitives)
        // for reconstruction, so there's no preparation needed.
      }
    };

    // struct ComputeGhostAndSubcellData {
    //   template <typename DbTagsList>
    //   static void apply(
    //       const gsl::not_null<std::vector<double>*> ghost_and_subcell_data,
    //       const db::DataBox<DbTagsList>& box,
    //       const Direction<volume_dim>& direction,
    //       const OrientationMap<volume_dim>& orientation) noexcept {}
    // };

    // struct CleanupAfterGhostAndSubcellData {
    //   template <typename DbTagsList>
    //   static void apply(
    //       const gsl::not_null<db::DataBox<DbTagsList>*> /*box*/) noexcept {
    //     // No memory to cleanup/free after sending the DG-subcell data
    //   }
    // };
  };

  using step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
      evolution::dg::Actions::ComputeTimeDerivative<EvolutionMetavars>,
      evolution::dg::Actions::ApplyBoundaryCorrections<EvolutionMetavars>,
      tmpl::conditional_t<
          local_time_stepping, tmpl::list<>,
          tmpl::list<Actions::RecordTimeStepperData<>, Actions::UpdateU<>>>,
      // TODO: TCI and recomputation here. Need to somehow gracefully transition
      // to FD without changing values at boundary. This means we can do:
      // 2. higher order FD, but slightly violating conservation
      //
      // 1. run TCI
      // 2. project to SCL. This projection "needs" to be conservative
      // 3. use Labels::BeginSubcellAfterSend to jump to next SCL action
      // Note: SCL in this case needs to project G to the SCL grid for use on
      //       the faces because that's the boundary correction we sent.
      evolution::dg::subcell::Actions::TciAndRollback,
      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      evolution::dg::subcell::Actions::SendDataForReconstruction<
          EvolutionMetavars>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<volume_dim>,
      Actions::Label<
          evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,
      evolution::dg::subcell::fd::Actions::TakeTimeStep,
      Actions::RecordTimeStepperData<>, Actions::UpdateU<>,
      evolution::dg::subcell::Actions::TciAndSwitchToDg,

      Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;

  enum class Phase {
    Initialization,
    LoadBalancing,
    WriteCheckpoint,
    RegisterWithObserver,
    InitializeTimeStepperHistory,
    Evolve,
    Exit
  };

  static std::string phase_name(Phase phase) noexcept {
    if (phase == Phase::LoadBalancing) {
      return "LoadBalancing";
    }
    ERROR(
        "Passed phase that should not be used in input file. Integer "
        "corresponding to phase is: "
        << static_cast<int>(phase));
  }

  using phase_changes =
      tmpl::list<PhaseControl::Registrars::VisitAndReturn<EvolutionMetavars,
                                                          Phase::LoadBalancing>,
                 PhaseControl::Registrars::CheckpointAndExitAfterWallclock<
                     EvolutionMetavars>>;

  using initialize_phase_change_decision_data =
      PhaseControl::InitializePhaseChangeDecisionData<phase_changes>;

  using phase_change_tags_and_combines_list =
      PhaseControl::get_phase_change_tags<phase_changes>;

  using const_global_cache_tags =
      tmpl::list<initial_data_tag, Tags::EventsAndTriggers,
                 PhaseControl::Tags::PhaseChangeAndTriggers<phase_changes>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::list<
      Actions::SetupDataBox,
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<1>,
      Initialization::Actions::ConservativeSystem<system>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<1, Frame::Logical>>,
      evolution::dg::subcell::Actions::Initialize<EvolutionMetavars>,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      tmpl::conditional_t<
          evolution::is_analytic_solution_v<initial_data>,
          Initialization::Actions::AddComputeTags<
              tmpl::list<evolution::Tags::AnalyticCompute<
                  1, initial_data_tag, analytic_solution_fields>>>,
          tmpl::list<>>,
      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      // Initialization::Actions::Minmod<1>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<Phase, Phase::RegisterWithObserver,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Phase, Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,

          Parallel::PhaseActions<
              Phase, Phase::Evolve,
              tmpl::list<Actions::RunEventsAndTriggers,
                         Actions::ChangeSlabSize, step_actions,
                         Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange<
                             phase_changes>>>>>;

  template <typename ParallelComponent>
  struct registration_list {
    using type = std::conditional_t<
        std::is_same_v<ParallelComponent, dg_element_array>,
        dg_registration_list, tmpl::list<>>;
  };

  using component_list =
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 dg_element_array>;

  static constexpr Options::String help{
      "Evolve the Burgers equation.\n\n"
      "The analytic solution is: Linear\n"
      "The numerical flux is:    LocalLaxFriedrichs\n"};

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<tuples::TaggedTuple<Tags...>*>
          phase_change_decision_data,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<EvolutionMetavars>&
          cache_proxy) noexcept {
    const auto next_phase =
        PhaseControl::arbitrate_phase_change<phase_changes>(
            phase_change_decision_data, current_phase,
            *(cache_proxy.ckLocalBranch()));
    if (next_phase.has_value()) {
      return next_phase.value();
    }
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &Burgers::BoundaryConditions::register_derived_with_charm,
    &Burgers::BoundaryCorrections::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
