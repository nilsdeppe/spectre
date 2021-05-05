#include <iostream>

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/Conservative/UpdateConservatives.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Minmod.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/NoSource.hpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianEuler/KhInstability.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/LaneEmdenStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/RiemannProblem.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
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
#include "Utilities/Functional.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/DgSubcell/Actions/TciAndRollback.hpp"
#include "Evolution/DgSubcell/Actions/TciAndSwitchToDg.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/ComputeBoundaryTerms.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/Events/ObserveFields.hpp"
#include "Evolution/DgSubcell/FaceLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/PrimsAfterRollback.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/ResizeAndComputePrimitives.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/TciOnDgGrid.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/TciOnFdGrid.hpp"
#include "NumericalAlgorithms/FiniteDifference/Minmod.hpp"

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

template <size_t Dim, typename InitialData>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = Dim;
  static constexpr dg::Formulation dg_formulation =
      dg::Formulation::StrongInertial;

  using initial_data = InitialData;
  static_assert(
      evolution::is_analytic_data_v<initial_data> xor
          evolution::is_analytic_solution_v<initial_data>,
      "initial_data must be either an analytic_data or an analytic_solution");

  using equation_of_state_type = typename initial_data::equation_of_state_type;

  using source_term_type = typename initial_data::source_term_type;

  using system =
      NewtonianEuler::System<Dim, equation_of_state_type, initial_data>;

  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;

  using initial_data_tag =
      tmpl::conditional_t<evolution::is_analytic_solution_v<initial_data>,
                          Tags::AnalyticSolution<initial_data>,
                          Tags::AnalyticData<initial_data>>;

  using boundary_condition_tag = initial_data_tag;
  using analytic_variables_tags =
      typename system::primitive_variables_tag::tags_list;

  using equation_of_state_tag =
      hydro::Tags::EquationOfState<equation_of_state_type>;

  using source_term_tag = NewtonianEuler::Tags::SourceTerm<initial_data>;
  static constexpr bool has_source_terms =
      not std::is_same_v<source_term_type, NewtonianEuler::Sources::NoSource>;

  using limiter = Tags::Limiter<NewtonianEuler::Limiters::Minmod<Dim>>;

  using time_stepper_tag = Tags::TimeStepper<
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>>;

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
                    Dim, ::Tags::Time,
                    tmpl::append<
                        typename system::variables_tag::tags_list,
                        typename system::primitive_variables_tag::tags_list,
                        tmpl::list<evolution::dg::subcell::Tags::TciStatus>>,
                    tmpl::conditional_t<
                        evolution::is_analytic_solution_v<initial_data>,
                        typename system::primitive_variables_tag::tags_list,
                        tmpl::list<>>>,
                Events::time_events<system>>>>,
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
    // Conservative vars tags
    using MassDensityCons = NewtonianEuler::Tags::MassDensityCons;
    using EnergyDensity = NewtonianEuler::Tags::EnergyDensity;
    using MomentumDensity = NewtonianEuler::Tags::MomentumDensity<volume_dim>;

    // Primitive vars tags
    using MassDensity = NewtonianEuler::Tags::MassDensity<DataVector>;
    using Velocity = NewtonianEuler::Tags::Velocity<DataVector, volume_dim>;
    using SpecificInternalEnergy =
        NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>;
    using Pressure = NewtonianEuler::Tags::Pressure<DataVector>;

    using evolved_vars_tags =
        tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>;
    using prim_tags =
        tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>;
    using prims_to_reconstruct_tags =
        tmpl::list<MassDensity, Velocity, Pressure>;
    using fluxes_tags = db::wrap_tags_in<::Tags::Flux, evolved_vars_tags,
                                         tmpl::size_t<Dim>, Frame::Inertial>;

    static constexpr bool subcell_enabled = true;
    // We send `ghost_zone_size` cell-centered grid points for variable
    // reconstruction, of which we need `ghost_zone_size-1` for reconstruction
    // to the internal side of the element face, and `ghost_zone_size` for
    // reconstruction to the external side of the element face.
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) noexcept {
      return db::get<NewtonianEuler::fd::Tags::Reconstructor<Dim>>(box)
          .ghost_zone_size();
    }

    struct TimeDerivative {
      // Things that need updating for 2d:
      // - normal vectors and their magnitude on the face
      // - det jacobian on the face and in the cell center
      // - the number of points of `vars_on_?_face` and `?_packaged_data`
      // - accounting for the DG correction sent from our neighbor and that we
      //   sent to our neighbor. This needs projection.
      // - the finite difference derivatives when computing the time derivative
      //   need to work correctly in the eta (and zeta) directions.
      // - we need to zero the time derivative before computing the
      //   xi-direction so we can always use +=

      template <typename DbTagsList>
      static void apply(
          const gsl::not_null<db::DataBox<DbTagsList>*> box,
          const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Grid>&
              cell_centered_logical_to_grid_inv_jacobian,
          const Scalar<
              DataVector>& /*cell_centered_det_inv_jacobian*/) noexcept {
        // static_assert(volume_dim == 1,
        //               "Currently subcell solver only implemented in 1d");

        const Mesh<volume_dim>& subcell_mesh =
            db::get<evolution::dg::subcell::Tags::Mesh<volume_dim>>(*box);
        const size_t num_pts = subcell_mesh.number_of_grid_points();
        const size_t reconstructed_num_pts =
            (subcell_mesh.extents(0) + 1) *
            subcell_mesh.extents().slice_away(0).product();

        const auto& cell_centered_logical_coords =
            db::get<evolution::dg::subcell::Tags::Coordinates<volume_dim,
                                                              Frame::Logical>>(
                *box);
        // Note: assumes
        std::array<double, volume_dim> one_over_delta_xi{};
        for (size_t i = 0; i < volume_dim; ++i) {
          // Note: assumes isotropic extents
          gsl::at(one_over_delta_xi, i) =
              1.0 / (cell_centered_logical_coords.get(0)[1] -
                     cell_centered_logical_coords.get(0)[0]);
        }

        const auto& recons =
            db::get<NewtonianEuler::fd::Tags::Reconstructor<Dim>>(*box);

        using flux_tags =
            typename NewtonianEuler::ComputeFluxes<Dim>::return_tags;

        const auto& element = db::get<domain::Tags::Element<volume_dim>>(*box);
        ASSERT(
            element.external_boundaries().size() == 0,
            "Can't have external boundaries right now with subcell. ElementID "
                << element.id());

        // Now package the data and compute the correction
        const auto& boundary_correction =
            db::get<evolution::Tags::BoundaryCorrection<system>>(*box);
        using derived_boundary_corrections = typename std::decay_t<decltype(
            boundary_correction)>::creatable_classes;
        std::array<Variables<evolved_vars_tags>, volume_dim>
            boundary_corrections{};
        tmpl::for_each<derived_boundary_corrections>(
            [&](auto derived_correction_v) noexcept {
              using DerivedCorrection =
                  tmpl::type_from<decltype(derived_correction_v)>;
              if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
                using dg_package_data_temporary_tags =
                    typename DerivedCorrection::dg_package_data_temporary_tags;
                static_assert(
                    std::is_same_v<dg_package_data_temporary_tags,
                                   tmpl::list<>>,
                    "Cannot yet support temporary tags with DG packaged data.");
                using dg_package_data_argument_tags =
                    tmpl::append<evolved_vars_tags, prim_tags, fluxes_tags,
                                 dg_package_data_temporary_tags>;
                // Computed prims and cons on face via reconstruction
                auto vars_on_lower_face = make_array<volume_dim>(
                    Variables<dg_package_data_argument_tags>(
                        reconstructed_num_pts));
                auto vars_on_upper_face = make_array<volume_dim>(
                    Variables<dg_package_data_argument_tags>(
                        reconstructed_num_pts));

                // Compute fluxes on faces
                call_with_dynamic_type<
                    void, typename NewtonianEuler::fd::Reconstructor<
                              Dim>::creatable_classes>(
                    &recons, [&box, &vars_on_lower_face, &vars_on_upper_face](
                                 const auto& reconstructor) noexcept {
                      db::apply<typename std::decay_t<decltype(
                          *reconstructor)>::reconstruction_argument_tags>(
                          [&vars_on_lower_face, &vars_on_upper_face,
                           &reconstructor](const auto&... args) noexcept {
                            reconstructor->reconstruct(
                                make_not_null(&vars_on_lower_face),
                                make_not_null(&vars_on_upper_face), args...);
                          },
                          *box);
                    });

                using dg_package_field_tags =
                    typename DerivedCorrection::dg_package_field_tags;
                // Allocated outside for loop to reduce allocations
                Variables<dg_package_field_tags> upper_packaged_data{
                    reconstructed_num_pts};
                Variables<dg_package_field_tags> lower_packaged_data{
                    reconstructed_num_pts};

                for (size_t i = 0; i < Dim; ++i) {
                  auto& vars_upper_face = gsl::at(vars_on_upper_face, i);
                  auto& vars_lower_face = gsl::at(vars_on_lower_face, i);
                  NewtonianEuler::ComputeFluxes<Dim>::apply(
                      make_not_null(
                          &get<tmpl::at_c<flux_tags, 0>>(vars_upper_face)),
                      make_not_null(
                          &get<tmpl::at_c<flux_tags, 1>>(vars_upper_face)),
                      make_not_null(
                          &get<tmpl::at_c<flux_tags, 2>>(vars_upper_face)),
                      get<MomentumDensity>(vars_upper_face),
                      get<EnergyDensity>(vars_upper_face),
                      get<Velocity>(vars_upper_face),
                      get<Pressure>(vars_upper_face));
                  NewtonianEuler::ComputeFluxes<Dim>::apply(
                      make_not_null(
                          &get<tmpl::at_c<flux_tags, 0>>(vars_lower_face)),
                      make_not_null(
                          &get<tmpl::at_c<flux_tags, 1>>(vars_lower_face)),
                      make_not_null(
                          &get<tmpl::at_c<flux_tags, 2>>(vars_lower_face)),
                      get<MomentumDensity>(vars_lower_face),
                      get<EnergyDensity>(vars_lower_face),
                      get<Velocity>(vars_lower_face),
                      get<Pressure>(vars_lower_face));

                  // Normal vectors are easy, flat space. Note that we use the
                  // sign convention on the normal vectors to be compatible with
                  // DG.
                  tnsr::i<DataVector, volume_dim, Frame::Inertial>
                      upper_outward_conormal{reconstructed_num_pts, 0.0};
                  upper_outward_conormal.get(i) = -1.0;
                  tnsr::i<DataVector, volume_dim, Frame::Inertial>
                      lower_outward_conormal{reconstructed_num_pts, 0.0};
                  lower_outward_conormal.get(i) = 1.0;

                  // Compute the packaged data
                  using dg_package_data_projected_tags =
                      tmpl::append<evolved_vars_tags, fluxes_tags,
                                   dg_package_data_temporary_tags,
                                   typename DerivedCorrection::
                                       dg_package_data_primitive_tags>;
                  evolution::dg::Actions::detail::dg_package_data<system>(
                      make_not_null(&upper_packaged_data),
                      dynamic_cast<const DerivedCorrection&>(
                          boundary_correction),
                      vars_upper_face, upper_outward_conormal, {std::nullopt},
                      *box,
                      typename DerivedCorrection::dg_package_data_volume_tags{},
                      dg_package_data_projected_tags{});

                  evolution::dg::Actions::detail::dg_package_data<system>(
                      make_not_null(&lower_packaged_data),
                      dynamic_cast<const DerivedCorrection&>(
                          boundary_correction),
                      vars_lower_face, lower_outward_conormal, {std::nullopt},
                      *box,
                      typename DerivedCorrection::dg_package_data_volume_tags{},
                      dg_package_data_projected_tags{});

                  // Now need to check if any of our neighbors are doing DG,
                  // because if so then we need to use whatever boundary data
                  // they sent instead of what we computed locally. Note: We
                  // could check this beforehand to avoid the extra work
                  //       of reconstruction and flux computations at the
                  //       boundaries.
                  evolution::dg::subcell::correct_package_data<true>(
                      make_not_null(&lower_packaged_data),
                      make_not_null(&upper_packaged_data), i, element,
                      subcell_mesh,
                      db::get<evolution::dg::Tags::MortarData<volume_dim>>(
                          *box));

                  // Compute the corrections on the faces. We only need to
                  // compute this once because we can just flip the normal
                  // vectors then
                  gsl::at(boundary_corrections, i)
                      .initialize(reconstructed_num_pts);
                  evolution::dg::subcell::compute_boundary_terms(
                      make_not_null(&gsl::at(boundary_corrections, i)),
                      dynamic_cast<const DerivedCorrection&>(
                          boundary_correction),
                      upper_packaged_data, lower_packaged_data);
                }
              }
            });

        // Now compute the actual time derivatives.
        using variables_tag = typename system::variables_tag;
        using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
        db::mutate<dt_variables_tag>(
            box, [&cell_centered_logical_to_grid_inv_jacobian, &num_pts,
                  &boundary_corrections, &subcell_mesh,
                  &one_over_delta_xi](const auto dt_vars_ptr) noexcept {
              dt_vars_ptr->initialize(num_pts, 0.0);
              auto& dt_mass = get<Tags::dt<MassDensityCons>>(*dt_vars_ptr);
              auto& dt_momentum = get<Tags::dt<MomentumDensity>>(*dt_vars_ptr);
              auto& dt_energy = get<Tags::dt<EnergyDensity>>(*dt_vars_ptr);

              for (size_t dim = 0; dim < Dim; ++dim) {
                Scalar<DataVector>& mass_density_correction =
                    get<MassDensityCons>(gsl::at(boundary_corrections, dim));
                tnsr::I<DataVector, volume_dim, Frame::Inertial>&
                    momentum_density_correction = get<MomentumDensity>(
                        gsl::at(boundary_corrections, dim));
                Scalar<DataVector>& energy_density_correction =
                    get<EnergyDensity>(gsl::at(boundary_corrections, dim));

                evolution::dg::subcell::add_cartesian_flux_divergence(
                    make_not_null(&get(dt_mass)),
                    gsl::at(one_over_delta_xi, dim),
                    cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                    get(mass_density_correction), subcell_mesh.extents(), dim);
                evolution::dg::subcell::add_cartesian_flux_divergence(
                    make_not_null(&get(dt_energy)),
                    gsl::at(one_over_delta_xi, dim),
                    cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                    get(energy_density_correction), subcell_mesh.extents(),
                    dim);
                for (size_t d = 0; d < volume_dim; ++d) {
                  evolution::dg::subcell::add_cartesian_flux_divergence(
                      make_not_null(&dt_momentum.get(d)),
                      gsl::at(one_over_delta_xi, dim),
                      cell_centered_logical_to_grid_inv_jacobian.get(dim, dim),
                      momentum_density_correction.get(d),
                      subcell_mesh.extents(), dim);
                }
              }
            });
      }
    };

    struct DgComputeSubcellNeighborPackagedData {
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
        const Mesh<volume_dim>& dg_mesh =
            db::get<domain::Tags::Mesh<volume_dim>>(box);
        // Since the active grid is DG, we need to use the inactive grid for
        // reconstruction.
        //
        // Note: since for NewtonianEuler prim recovery is basically free, so we
        // could do either the prim recovery on the subcells or project the
        // prims. Since for relativistic hydro the prims are expensive to
        // recovery, we do the projection.
        const auto volume_prims = evolution::dg::subcell::fd::project(
            db::get<typename system::primitive_variables_tag>(box), dg_mesh,
            subcell_mesh.extents());

        const auto& recons =
            db::get<NewtonianEuler::fd::Tags::Reconstructor<Dim>>(box);
        const auto& boundary_correction =
            db::get<evolution::Tags::BoundaryCorrection<system>>(box);
        using derived_boundary_corrections = typename std::decay_t<decltype(
            boundary_correction)>::creatable_classes;
        tmpl::for_each<derived_boundary_corrections>(
            [&box, &boundary_correction, &dg_mesh, &mortars_to_reconstruct_to,
             &nhbr_package_data, &nhbr_subcell_data, &recons, &subcell_mesh,
             &volume_prims](auto derived_correction_v) noexcept {
              using DerivedCorrection =
                  tmpl::type_from<decltype(derived_correction_v)>;
              if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
                using dg_package_data_temporary_tags =
                    typename DerivedCorrection::dg_package_data_temporary_tags;
                static_assert(
                    std::is_same_v<dg_package_data_temporary_tags,
                                   tmpl::list<>>,
                    "Cannot yet support temporary tags with DG packaged data.");
                using dg_package_data_argument_tags =
                    tmpl::append<evolved_vars_tags, prim_tags, fluxes_tags,
                                 dg_package_data_temporary_tags>;

                const auto& element =
                    db::get<domain::Tags::Element<volume_dim>>(box);
                const auto& eos = get<hydro::Tags::EquationOfStateBase>(box);

                for (const auto& mortar_id : mortars_to_reconstruct_to) {
                  // Computed prims and cons on face via reconstruction
                  // ASSERT(volume_dim == 1, "The size assumes 1d");
                  const size_t num_face_pts =
                      subcell_mesh.extents()
                          .slice_away(mortar_id.first.dimension())
                          .product();
                  Variables<dg_package_data_argument_tags> vars_on_face{
                      num_face_pts, 0.0};

                  call_with_dynamic_type<
                      void, typename NewtonianEuler::fd::Reconstructor<
                                Dim>::creatable_classes>(
                      &recons, [&element, &eos, &mortar_id, &nhbr_subcell_data,
                                &subcell_mesh, &vars_on_face, &volume_prims](
                                   const auto& reconstructor) noexcept {
                        reconstructor->reconstruct_fd_neighbor(
                            make_not_null(&vars_on_face), volume_prims, eos,
                            element, nhbr_subcell_data, subcell_mesh,
                            mortar_id.first);
                      });

                  using flux_tags = typename NewtonianEuler::ComputeFluxes<
                      volume_dim>::return_tags;
                  NewtonianEuler::ComputeFluxes<volume_dim>::apply(
                      make_not_null(
                          &get<tmpl::at_c<flux_tags, 0>>(vars_on_face)),
                      make_not_null(
                          &get<tmpl::at_c<flux_tags, 1>>(vars_on_face)),
                      make_not_null(
                          &get<tmpl::at_c<flux_tags, 2>>(vars_on_face)),
                      get<MomentumDensity>(vars_on_face),
                      get<EnergyDensity>(vars_on_face),
                      get<Velocity>(vars_on_face), get<Pressure>(vars_on_face));

                  tnsr::i<DataVector, volume_dim, Frame::Inertial>
                      normal_covector = get<
                          evolution::dg::Tags::NormalCovector<volume_dim>>(
                          *db::get<evolution::dg::Tags::
                                       NormalCovectorAndMagnitude<volume_dim>>(
                               box)
                               .at(mortar_id.first));
                  for (auto& t : normal_covector) {
                    t *= -1.0;
                  }
                  if constexpr (Dim > 1) {
                    const auto dg_normal_covector = normal_covector;
                    for (size_t i = 0; i < Dim; ++i) {
                      normal_covector.get(i) =
                          evolution::dg::subcell::fd::project(
                              dg_normal_covector.get(i),
                              dg_mesh.slice_away(mortar_id.first.dimension()),
                              subcell_mesh.extents().slice_away(
                                  mortar_id.first.dimension()));
                    }
                  }

                  // Compute the packaged data
                  using dg_package_field_tags =
                      typename DerivedCorrection::dg_package_field_tags;
                  Variables<dg_package_field_tags> packaged_data{num_face_pts};
                  using dg_package_data_projected_tags =
                      tmpl::append<evolved_vars_tags, fluxes_tags,
                                   dg_package_data_temporary_tags,
                                   typename DerivedCorrection::
                                       dg_package_data_primitive_tags>;
                  evolution::dg::Actions::detail::dg_package_data<system>(
                      make_not_null(&packaged_data),
                      dynamic_cast<const DerivedCorrection&>(
                          boundary_correction),
                      vars_on_face, normal_covector, {std::nullopt}, box,
                      typename DerivedCorrection::dg_package_data_volume_tags{},
                      dg_package_data_projected_tags{});
                  if constexpr (volume_dim == 1) {
                    (void)dg_mesh;
                    nhbr_package_data[mortar_id] = std::vector<double>{
                        packaged_data.data(),
                        packaged_data.data() + packaged_data.size()};
                  } else {
                    // Reconstruct the DG solution.
                    // Really we should be solving the boundary correction and
                    // then reconstructing, but away from a shock this doesn't
                    // matter.
                    auto dg_packaged_data =
                        evolution::dg::subcell::fd::reconstruct(
                            packaged_data,
                            dg_mesh.slice_away(mortar_id.first.dimension()),
                            subcell_mesh.extents().slice_away(
                                mortar_id.first.dimension()));
                    nhbr_package_data[mortar_id] = std::vector<double>{
                        dg_packaged_data.data(),
                        dg_packaged_data.data() + dg_packaged_data.size()};
                  }
                }
              }
            });
        return nhbr_package_data;
      }
    };

    using GhostDataToSlice =
        NewtonianEuler::subcell::PrimitiveGhostDataToSlice<Dim>;
  };

  using step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
      evolution::dg::Actions::ComputeTimeDerivative<EvolutionMetavars>,
      evolution::dg::Actions::ApplyBoundaryCorrections<EvolutionMetavars>,
      tmpl::conditional_t<
          local_time_stepping, tmpl::list<>,
          tmpl::list<Actions::RecordTimeStepperData<>, Actions::UpdateU<>>>,
      // Note: The primitive variables are computed as part of the TCI.
      evolution::dg::subcell::Actions::TciAndRollback<
          NewtonianEuler::subcell::TciOnDgGrid<Dim>>,
      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      evolution::dg::subcell::Actions::SendDataForReconstruction<
          volume_dim,
          NewtonianEuler::subcell::PrimitiveGhostDataOnSubcells<Dim>>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<volume_dim>,
      Actions::Label<
          evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,
      Actions::MutateApply<NewtonianEuler::subcell::PrimsAfterRollback<Dim>>,
      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          typename SubcellOptions::TimeDerivative>,
      Actions::RecordTimeStepperData<>, Actions::UpdateU<>,
      evolution::dg::subcell::Actions::TciAndSwitchToDg<
          NewtonianEuler::subcell::TciOnFdGrid<Dim>>,
      Actions::MutateApply<NewtonianEuler::subcell::ResizeAndComputePrims<Dim>>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;

  enum class Phase {
    Initialization,
    InitializeTimeStepperHistory,
    RegisterWithObserver,
    LoadBalancing,
    WriteCheckpoint,
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

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::list<
      Actions::SetupDataBox,
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<Dim>,
      Initialization::Actions::ConservativeSystem<system,
                                                  equation_of_state_tag>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<Dim, Frame::Logical>>,
      Actions::UpdateConservatives,
      evolution::dg::subcell::Actions::Initialize<
          Dim, system, NewtonianEuler::subcell::DgInitialDataTci<Dim>>,
      Actions::UpdateConservatives,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      Initialization::Actions::AddComputeTags<
          tmpl::list<NewtonianEuler::Tags::SoundSpeedSquaredCompute<DataVector>,
                     NewtonianEuler::Tags::SoundSpeedCompute<DataVector>>>,
      tmpl::conditional_t<
          evolution::is_analytic_solution_v<initial_data>,
          Initialization::Actions::AddComputeTags<
              tmpl::list<evolution::Tags::AnalyticCompute<
                  Dim, initial_data_tag, analytic_variables_tags>>>,
          tmpl::list<>>,
      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      // Initialization::Actions::Minmod<Dim>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<
              Phase, Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,

          Parallel::PhaseActions<Phase, Phase::RegisterWithObserver,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Phase, Phase::Evolve,
              tmpl::list<Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions, Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange<
                             phase_changes>>>>>;

  template <typename ParallelComponent>
  struct registration_list {
    using type =
        std::conditional_t<std::is_same_v<ParallelComponent, dg_element_array>,
                           dg_registration_list, tmpl::list<>>;
  };

  using component_list =
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 dg_element_array>;

  using const_global_cache_tags = tmpl::list<
      NewtonianEuler::fd::Tags::Reconstructor<Dim>, initial_data_tag,
      tmpl::conditional_t<has_source_terms, source_term_tag, tmpl::list<>>,
      Tags::EventsAndTriggers,
      PhaseControl::Tags::PhaseChangeAndTriggers<phase_changes>>;

  static constexpr Options::String help{
      "Evolve the Newtonian Euler system in conservative form.\n\n"};

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
            "Unknown type of phase. Did you static_cast<Phase> to an integral "
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
    &NewtonianEuler::BoundaryConditions::register_derived_with_charm,
    &NewtonianEuler::BoundaryCorrections::register_derived_with_charm,
    &NewtonianEuler::fd::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
