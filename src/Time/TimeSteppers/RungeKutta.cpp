// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta.hpp"

#include <algorithm>

#include "Time/EvolutionOrdering.hpp"
#include "Time/History.hpp"
#include "Time/Time.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Math.hpp"

namespace TimeSteppers {

uint64_t RungeKutta::number_of_substeps() const {
  return butcher_tableau().result_coefficients.size();
}

uint64_t RungeKutta::number_of_substeps_for_error() const {
  return std::max(butcher_tableau().result_coefficients.size(),
                  butcher_tableau().error_coefficients.size());
}

size_t RungeKutta::number_of_past_steps() const { return 0; }

namespace {
TimeStepId next_time_id_from_substeps(
    const TimeStepId& current_id, const TimeDelta& time_step,
    const std::vector<double>& substep_times,
    const size_t number_of_substeps) {
  ASSERT(substep_times.size() + 1 >= number_of_substeps,
         "More result coefficients than substeps");
  const auto substep = current_id.substep();

  if (substep >= number_of_substeps) {
    ERROR("In substep should be less than the number of steps, not "
          << substep << "/" << number_of_substeps);
  } else if (substep == number_of_substeps - 1) {
    return current_id.next_step(time_step);
  } else {
    return current_id.next_substep(time_step, substep_times[substep]);
  }
}
}  // namespace

TimeStepId RungeKutta::next_time_id(const TimeStepId& current_id,
                                    const TimeDelta& time_step) const {
  return next_time_id_from_substeps(current_id, time_step,
                                    butcher_tableau().substep_times,
                                    number_of_substeps());
}

TimeStepId RungeKutta::next_time_id_for_error(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  return next_time_id_from_substeps(current_id, time_step,
                                    butcher_tableau().substep_times,
                                    number_of_substeps_for_error());
}

namespace {
template <typename T>
void compute_substep(const gsl::not_null<T*> u,
                     const ConstUntypedHistory<T>& history, const double dt,
                     const std::vector<double>& substep_coefficients) {
  if (substep_coefficients[0] != 0.0) {
    *u = *history.front().value +
         substep_coefficients[0] * dt * history.front().derivative;
  } else {
    *u = *history.front().value;
  }
  for (size_t i = 1; i < substep_coefficients.size(); ++i) {
    if (substep_coefficients[i] != 0.0) {
      *u += substep_coefficients[i] * dt * history.substeps()[i - 1].derivative;
    }
  }
}

template <typename T>
void update_u_impl_with_tableau(const gsl::not_null<T*> u,
                                const MutableUntypedHistory<T>& history,
                                const TimeDelta& time_step,
                                const RungeKutta::ButcherTableau& tableau,
                                const size_t number_of_substeps) {
  if constexpr (std::is_same_v<T, DataVector>) {
    __builtin_prefetch(u->data(), 1, 3);
  }

  // Clean up old history
  if (history.at_step_start()) {
    history.clear_substeps();
    if (history.size() > 1) {
      history.pop_front();
    }
  } else {
    if constexpr (std::is_same_v<T, DataVector>) {
      __builtin_prefetch(history.front().value->data(), 0, 1);
      __builtin_prefetch(history.front().derivative.data(), 0, 1);
    }
    history.discard_value(history.substeps().back().time_step_id);
  }
  ASSERT(history.size() == 1, "Have more than one step after cleanup.");

  const double dt = time_step.value();

  const auto substep = history.substeps().size();
  if (substep == number_of_substeps - 1) {
    compute_substep(u, history, dt, tableau.result_coefficients);
  } else if (substep < number_of_substeps - 1) {
    compute_substep(u, history, dt, tableau.substep_coefficients[substep]);
  } else {
    ERROR("Substep should be less than " << number_of_substeps << ", not "
                                         << substep);
  }
}
}  // namespace

template <typename T>
void RungeKutta::update_u_impl(const gsl::not_null<T*> u,
                               const MutableUntypedHistory<T>& history,
                               const TimeDelta& time_step) const {
  ASSERT(history.integration_order() == order(),
         "Fixed-order stepper cannot run at order "
             << history.integration_order());
  return update_u_impl_with_tableau(u, history, time_step, butcher_tableau(),
                                    number_of_substeps());
}

template <typename T>
bool RungeKutta::update_u_impl(const gsl::not_null<T*> u,
                               const gsl::not_null<T*> u_error,
                               const MutableUntypedHistory<T>& history,
                               const TimeDelta& time_step) const {
  ASSERT(history.integration_order() == order(),
         "Fixed-order stepper cannot run at order "
             << history.integration_order());

  const auto& tableau = butcher_tableau();
  const auto number_of_substeps = number_of_substeps_for_error();
  update_u_impl_with_tableau(u, history, time_step, tableau,
                             number_of_substeps);

  const size_t substep = history.substeps().size();

  if (substep < number_of_substeps - 1) {
    return false;
  }

  const double dt = time_step.value();
  compute_substep(u_error, history, dt, tableau.error_coefficients);
  *u_error = *u - *u_error;

  return true;
}

template <typename T>
bool RungeKutta::dense_update_u_impl(const gsl::not_null<T*> u,
                                     const ConstUntypedHistory<T>& history,
                                     const double time) const {
  if (not history.at_step_start()) {
    return false;
  }
  const double step_start = history.front().time_step_id.step_time().value();
  const double step_end = history.back().time_step_id.step_time().value();
  if (time == step_end) {
    // Special case necessary for dense output at the initial time,
    // before taking a step.
    *u = *history.back().value;
    return true;
  }
  const evolution_less<double> before{step_end > step_start};
  if (history.size() == 1 or before(step_end, time)) {
    return false;
  }
  const double step_size = step_end - step_start;
  const double output_fraction = (time - step_start) / step_size;
  ASSERT(output_fraction >= 0.0, "Attempting dense output at time "
                                     << time << ", but already progressed past "
                                     << step_start);

  const auto& tableau = butcher_tableau();

  *u = *history.front().value;
  const auto number_of_dense_coefficients = tableau.dense_coefficients.size();
  const size_t number_of_substep_terms = std::min(
      tableau.result_coefficients.size(), number_of_dense_coefficients);
  for (size_t i = 0; i < number_of_substep_terms; ++i) {
    const double coef =
        evaluate_polynomial(tableau.dense_coefficients[i], output_fraction);
    if (coef != 0.0) {
      *u += coef * step_size *
            (i == 0 ? history.front() : history.substeps()[i - 1]).derivative;
    }
  }

  if (number_of_dense_coefficients > number_of_substep_terms) {
    // We use the derivative at the end of the step.
    const double coef =
        evaluate_polynomial(tableau.dense_coefficients.back(), output_fraction);
    if (coef != 0.0) {
      *u += coef * step_size * history.back().derivative;
    }
  }

  return true;
}

template <typename T>
bool RungeKutta::can_change_step_size_impl(
    const TimeStepId& time_id,
    const ConstUntypedHistory<T>& /*history*/) const {
  return time_id.substep() == 0;
}

TIME_STEPPER_DEFINE_OVERLOADS(RungeKutta)
}  // namespace TimeSteppers
