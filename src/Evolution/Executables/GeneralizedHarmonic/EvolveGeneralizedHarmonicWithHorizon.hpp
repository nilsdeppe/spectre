// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolate.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "Options/FactoryHelpers.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepControllers/Factory.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

// First template parameter specifies the source of the initial data, which
// could be an analytic solution, analytic data, or imported numerical data.
// Second template parameter specifies the analytic solution used when imposing
// dirichlet boundary conditions or against which to compute error norms.
template <typename InitialData, typename BoundaryConditions>
struct EvolutionMetavars
    : public virtual GeneralizedHarmonicDefaults,
      public GeneralizedHarmonicTemplateBase<
          EvolutionMetavars<InitialData, BoundaryConditions>> {
  static constexpr Options::String help{
      "Evolve the Einstein field equations using the Generalized Harmonic "
      "formulation,\n"
      "on a domain with a single horizon and corresponding excised region"};

  struct AhA {
    using tags_to_observe =
        tmpl::list<StrahlkorperGr::Tags::AreaCompute<frame>,
                   StrahlkorperGr::Tags::IrreducibleMassCompute<frame>>;
    using compute_items_on_source = tmpl::list<
        gr::Tags::SpatialMetricCompute<volume_dim, frame, DataVector>,
        ah::Tags::InverseSpatialMetricCompute<volume_dim, frame>,
        ah::Tags::ExtrinsicCurvatureCompute<volume_dim, frame>,
        ah::Tags::SpatialChristoffelSecondKindCompute<volume_dim, frame>>;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
                   gr::Tags::InverseSpatialMetric<volume_dim, frame>,
                   gr::Tags::ExtrinsicCurvature<volume_dim, frame>,
                   gr::Tags::SpatialChristoffelSecondKind<volume_dim, frame>>;
    using compute_items_on_target = tmpl::append<
        tmpl::list<StrahlkorperGr::Tags::AreaElementCompute<frame>>,
        tags_to_observe>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<AhA, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<AhA, ::Frame::Inertial>;
    using horizon_find_failure_callback =
        intrp::callbacks::ErrorOnFailedApparentHorizon;
    using post_horizon_find_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, AhA, AhA>;
  };

  using interpolation_target_tags = tmpl::list<AhA>;
  using interpolator_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Pi<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Phi<volume_dim, frame>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = Options::add_factory_classes<
        typename GeneralizedHarmonicTemplateBase<
            EvolutionMetavars>::factory_creation::factory_classes,
        tmpl::pair<Event, tmpl::list<intrp::Events::Interpolate<
                              3, AhA, interpolator_source_vars>>>>;
  };

  using phase_changes = typename GeneralizedHarmonicTemplateBase<
      EvolutionMetavars>::phase_changes;

  using const_global_cache_tags = tmpl::list<
      typename GeneralizedHarmonicTemplateBase<
          EvolutionMetavars>::analytic_solution_tag,
      Tags::EventsAndTriggers,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma0<
          volume_dim, frame>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma1<
          volume_dim, frame>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma2<
          volume_dim, frame>,
      PhaseControl::Tags::PhaseChangeAndTriggers<phase_changes>>;

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          typename AhA::post_horizon_find_callback>>;

  using dg_registration_list =
      tmpl::push_back<typename GeneralizedHarmonicTemplateBase<
                          EvolutionMetavars>::dg_registration_list,
                      intrp::Actions::RegisterElementWithInterpolator>;

  template <typename ParallelComponent>
  struct registration_list {
    using type = std::conditional_t<
        std::is_same_v<ParallelComponent,
                       typename GeneralizedHarmonicTemplateBase<
                           EvolutionMetavars>::gh_dg_element_array>,
        dg_registration_list, tmpl::list<>>;
  };

  using component_list =
      tmpl::push_back<typename GeneralizedHarmonicTemplateBase<
                          EvolutionMetavars>::component_list,
                      intrp::Interpolator<EvolutionMetavars>,
                      intrp::InterpolationTarget<EvolutionMetavars, AhA>>;
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &disable_openblas_multithreading,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &GeneralizedHarmonic::BoundaryConditions::register_derived_with_charm,
    &GeneralizedHarmonic::BoundaryCorrections::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &GeneralizedHarmonic::ConstraintDamping::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
