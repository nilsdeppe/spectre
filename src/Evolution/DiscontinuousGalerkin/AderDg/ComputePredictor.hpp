// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/AderDg/ConstantGuess.hpp"
#include "Evolution/AderDg/InitialGuesses.hpp"
#include "Evolution/AderDg/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace tuples {
template <typename...>
class TaggedTuple;  // IWYU pragma: keep
}  // namespace tuples
/// \endcond

/*!
 * \brief Contains everything necessary for SAMR-ADER-DG
 *
 * Implementation steps:
 * ~1. Figure out and write data structure for predictor solution~
 * ~2. Figure out interface and implement constant initial guess~
 * ~3. Implement new matrices~
 * 4. Implement Picard iteration for predictor
 * 5. Implement DG corrector
 * 6. Get a priori slope limiting to work
 * 7. Implement projection initial guess
 * 8. Implement CEM initial guess
 * 9. Implement CERK initial guess
 * 10. Implement resistive RMHD equations
 * 11. Work out stiff source term initial guesses.
 * 12. Implement cell-average initial guess
 * 13. Implement MUSCL-CN initial guess
 * 14. Implement CIM initial guess
 *
 * ## Predictor data structure
 * We want to be able to apply derivatives and matrices in general to the entire
 * spacetime predictor solution at once. This means we should arrange the data
 * in one contiguous block. The typical layout of a variables is
 * `x,y,z,component`, though we need to also include the time. The most obvious
 * options are `x,y,z,t,component` and `x,y,z,component,t`. For computing
 * derivatives the order of the last two is irrelevant, both will work just
 * fine. For apply the temporal stiffness matrix we need the order
 * `t,x,y,z,component`, so that does not strictly dictate a preference. We would
 * like to be able to apply the RHS at all spacetime grid points, which finally
 * leads us to the suggestion of `x,y,z,t,component`. We then must do a
 * transpose when apply then temporal inverse stiffness matrix.
 *
 * ## Corrector data structure
 * During the corrector phase we have to apply the matrix <phi_i, theta_a>,
 * which is effectively an integration in time.
 *
 */
namespace AderDg {
struct Predictor {
  using const_global_cache_tags = tmpl::list<OptionTags::AderDg>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::size<DbTagsList>::value != 0> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    using system = typename Metavariables::system;
    constexpr size_t volume_dim = system::volume_dim;
    using variables_tag = typename system::variables_tag;
    using corrector_variables_tag =
        db::add_tag_prefix<Tags::Corrector, variables_tag>;

    // Compute initial guess.
    const size_t number_of_temporal_grid_points =
        db::get<Tags::TemporalGridPoints>(box);
    const auto& mesh = db::get<::Tags::Mesh<volume_dim>>(box);
    const auto& aderdg_options = Parallel::get<OptionTags::AderDg>(cache);
    // Sets the tag ::Tags::Variables<predictor_variables_tag>
    set_initial_guess<corrector_variables_tag, variables_tag>(
        make_not_null(&box), aderdg_options, number_of_temporal_grid_points,
        mesh);

    // Perform Picard iteration to get solution.

    return std::forward_as_tuple(std::move(box));
  }

 private:
  template <typename CorrectorVariablesTag, typename VariablesTag,
            typename DbTagsList, size_t Dim>
  static void set_initial_guess(gsl::not_null<db::DataBox<DbTagsList>*> box,
                                const OptionTags::AderDg& options,
                                const size_t number_of_temporal_grid_points,
                                const Mesh<Dim>& mesh);

  template <typename PredictorVariablesTag, typename VariablesTag,
            typename DbTagsList, size_t Dim>
  static void perform_picard_iterations(
      gsl::not_null<db::DataBox<DbTagsList>*> box,
      const OptionTags::AderDg& options,
      const size_t number_of_temporal_grid_points, const Mesh<Dim>& mesh);
};

template <typename CorrectorVariablesTag, typename VariablesTag,
          typename DbTagsList, size_t Dim>
void Predictor::set_initial_guess(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const OptionTags::AderDg& options,
    const size_t number_of_temporal_grid_points, const Mesh<Dim>& mesh) {
  // TODO(nils): need to check if we actually have history data for fancier
  // initial guesses. If not we can use the constant initial guess as a
  // fallback, or the CERK initial guess.
  // InitialGuess::TypeOfGuess initial_guess type =
  switch (options.initial_guess_type()) {
    case InitialGuess::TypeOfGuess::Constant:
      db::mutate<tmpl::list<VariablesTag>>(
          box,
          [number_of_temporal_grid_points, &mesh](
              const auto ptr_predictor_vars, const auto& vars_at_initial_time) {
            InitialGuess::constant(ptr_predictor_vars, vars_at_initial_time,
                                   number_of_temporal_grid_points, mesh);
          },
          db::get<CorrectorVariablesTag>(*box));
      return;
    default:
      ERROR("Currently unsupported initial guess type: "
            << options.initial_guess_type());
  };
}

template <typename PredictorVariablesTag, typename VariablesTag,
          typename DbTagsList, size_t Dim>
void Predictor::perform_picard_iterations(
    gsl::not_null<db::DataBox<DbTagsList>*> box,
    const OptionTags::AderDg& options,
    const size_t number_of_temporal_grid_points, const Mesh<Dim>& mesh) {
  // perform Picard iterations until we have converged
  while (true) {
  }
}
}  // namespace AderDg
