// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"

#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveRecoveryData.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Simd/Simd.hpp"

namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes {

namespace {
// Equation (26)
template <typename T>
T compute_x(const T mu, const T b_squared) {
  using std::fma;
  const T one(1.0);
  return 1.0 / fma(mu, b_squared, one);
}

// Equation (38)
template <typename T>
T compute_r_bar_squared(const T mu, const T x, const T r_squared,
                        const T r_dot_b_squared) {
  return x * fma(r_squared, x, mu * (1.0 + x) * r_dot_b_squared);
}

// Equations (33) and (32)
template <typename T>
T compute_v_0_squared(const T r_squared, const T h_0_squared) {
  static T velocity_squared_upper_bound(
      1.0 - 4.0 * std::numeric_limits<double>::epsilon());
  using std::min;
  return min(r_squared / (h_0_squared + r_squared),
             velocity_squared_upper_bound);
}

template <typename T>
struct Primitives {
  const T rest_mass_density;
  const T lorentz_factor;
  const T pressure;
  const T specific_internal_energy;
  const T q_bar;
  const T r_bar_squared;
};

// Function to tighten master function bracket in corner case discussed in
// Appendix A when Equation (41) produces a rest mass density outside the
// valid range of the EOS
template <typename T>
class CornerCaseFunction {
 public:
  CornerCaseFunction(const T w_target, const T r_squared, const T b_squared,
                     const T r_dot_b_squared)
      : v_squared_target_(1.0 - 1.0 / square(w_target)),
        r_squared_(r_squared),
        b_squared_(b_squared),
        r_dot_b_squared_(r_dot_b_squared) {}

  T operator()(const T mu) const {
    const T x = compute_x(mu, b_squared_);
    const T r_bar_squared =
        compute_r_bar_squared(mu, x, r_squared_, r_dot_b_squared_);
    // v = mu*r_bar, see text after Equation (31)
    // target is v satisfying W(mu) = D/ rho_{min/max}
    return square(mu) * r_bar_squared - v_squared_target_;
  }

 private:
  const T v_squared_target_;
  const T r_squared_;
  const T b_squared_;
  const T r_dot_b_squared_;
};

// Function whose root is upper bracket of master function, see Sec. II.F
template <typename T>
class AuxiliaryFunction {
 public:
  AuxiliaryFunction(const T h_0_squared, const T r_squared, const T b_squared,
                    const T r_dot_b_squared)
      : h_0_squared_(h_0_squared),
        r_squared_(r_squared),
        b_squared_(b_squared),
        r_dot_b_squared_(r_dot_b_squared) {}

  T operator()(T mu) const {
    const T x = compute_x(mu, b_squared_);
    const T r_bar_squared =
        compute_r_bar_squared(mu, x, r_squared_, r_dot_b_squared_);
    // Equation (49)
    return mu * sqrt(h_0_squared_ + r_bar_squared) - 1.0;
  }

 private:
  const T h_0_squared_;
  const T r_squared_;
  const T b_squared_;
  const T r_dot_b_squared_;
};

// Master function, see Equation (44) in Sec. II.E
template <typename T, typename Eos>
class FunctionOfMu {
 public:
  FunctionOfMu(const T total_energy_density, const T momentum_density_squared,
               const T momentum_density_dot_magnetic_field,
               const T magnetic_field_squared,
               const T rest_mass_density_times_lorentz_factor,
               const T electron_fraction, const Eos& equation_of_state)
      : q_(total_energy_density / rest_mass_density_times_lorentz_factor - 1.0),
        r_squared_(momentum_density_squared /
                   square(rest_mass_density_times_lorentz_factor)),
        b_squared_(magnetic_field_squared /
                   rest_mass_density_times_lorentz_factor),
        r_dot_b_squared_(square(momentum_density_dot_magnetic_field) /
                         cube(rest_mass_density_times_lorentz_factor)),
        rest_mass_density_times_lorentz_factor_(
            rest_mass_density_times_lorentz_factor),
        electron_fraction_(electron_fraction),
        equation_of_state_(equation_of_state),
        h_0_(equation_of_state_.specific_enthalpy_lower_bound()),
        h_0_squared_(square(h_0_)),
        v_0_squared_(compute_v_0_squared(r_squared_, h_0_squared_)),
        one_over_rho_max_(equation_of_state_.rest_mass_density_upper_bound()),
        one_over_rho_min_(equation_of_state_.rest_mass_density_lower_bound()) {}

  std::pair<T, T> root_bracket(
      T rest_mass_density_times_lorentz_factor,
      get_value_type_or_default_t<T, T> absolute_tolerance,
      get_value_type_or_default_t<T, T> relative_tolerance,
      size_t max_iterations) const;

  Primitives<T> primitives(T mu) const;

  T operator()(const T mu) const;

 private:
  const T q_;
  const T r_squared_;
  const T b_squared_;
  const T r_dot_b_squared_;
  const T rest_mass_density_times_lorentz_factor_;
  const T electron_fraction_;
  const Eos& equation_of_state_;
  const T h_0_;
  const T h_0_squared_;
  const T v_0_squared_;
  const T one_over_rho_max_;
  const T one_over_rho_min_;
};

template <typename T, typename Eos>
std::pair<T, T> FunctionOfMu<T, Eos>::root_bracket(
    const T rest_mass_density_times_lorentz_factor,
    const get_value_type_or_default_t<T, T> absolute_tolerance,
    const get_value_type_or_default_t<T, T> relative_tolerance,
    const size_t max_iterations) const {
  // see text between Equations (49) and (50) and after Equation (54)
  T lower_bound(0.0);
  // We use `1 / (h_0_ + numeric_limits<T>::min())` to avoid division by
  // zero in a way that avoids conditionals.

  T upper_bound(1.0 / (h_0_ + std::numeric_limits<T>::min()));
  if (const auto mask = r_squared_ < h_0_squared_; xsimd::any(mask)) {
    // need to solve auxiliary function to determine mu_+ which will
    // be the upper bound for the master function bracket
    const auto auxiliary_function = AuxiliaryFunction{
        h_0_squared_, r_squared_, b_squared_, r_dot_b_squared_};
    if constexpr (std::is_fundamental_v<T>) {
      upper_bound = RootFinder::toms748(auxiliary_function, lower_bound,
                                        upper_bound, absolute_tolerance,
                                        relative_tolerance, max_iterations);
    } else {
      upper_bound = xsimd::select(
          mask,
          RootFinder::toms748(auxiliary_function, lower_bound, upper_bound,
                              absolute_tolerance, relative_tolerance,
                              max_iterations, not mask),
          upper_bound);
    }
  }

  // Determine if the corner case discussed in Appendix A occurs where the
  // mass density is outside the valid range of the EOS
  const T rho_min(equation_of_state_.rest_mass_density_lower_bound());
  const T rho_max(equation_of_state_.rest_mass_density_upper_bound());

  // If this is triggering, the most likely cause is that the density cutoff
  // for atmosphere is smaller than the minimum density of the EOS, i.e. this
  // point should have been flagged as atmosphere
  if (const auto mask = rest_mass_density_times_lorentz_factor < rho_min;
      simd::any(mask)) {
    throw std::runtime_error("Density too small for EOS");
  }

  const T mu_b = upper_bound;
  const T x = compute_x(mu_b, b_squared_);
  const T r_bar_squared =
      compute_r_bar_squared(mu_b, x, r_squared_, r_dot_b_squared_);
  // Equation (40)
  using std::min;
  const T v_hat_squared =
      min(square(mu_b) * r_bar_squared, v_0_squared_);
  // const T w_hat = 1.0 / sqrt(1.0 - v_hat_squared);
  const T one_over_w_hat = sqrt(1.0 - v_hat_squared);

  // If this is being triggered, the only possible recourse would be
  // to limit the rest mass density to the maximum value allowed by the
  // EOS.  This seems questionable, so it is treated as an inversion failure.
  // The exception should be caught by the try-catch in apply() which will
  // return std::nullopt
  // TODO: this shouldn't fail with an exception, we should return a mask of
  // values we couldn't solve for.
  if (const auto mask =
          rest_mass_density_times_lorentz_factor * one_over_w_hat > rho_max;
      xsimd::any(mask)) {
    throw std::runtime_error("Density too big for EOS");
  }

  if (const auto mask =
          rest_mass_density_times_lorentz_factor * one_over_w_hat < rho_min;
      xsimd::any(mask)) {
    // adjust lower bound
    const auto corner_case_function = CornerCaseFunction{
        rest_mass_density_times_lorentz_factor * one_over_rho_max_, r_squared_,
        b_squared_, r_dot_b_squared_};
    using std::max;
    if constexpr (std::is_fundamental_v<T>) {
      lower_bound = max(
          lower_bound, RootFinder::toms748(corner_case_function, lower_bound,
                                           upper_bound, absolute_tolerance,
                                           relative_tolerance, max_iterations));
    } else {
      lower_bound = max(
          lower_bound,
          RootFinder::toms748(corner_case_function, lower_bound, upper_bound,
                              absolute_tolerance, relative_tolerance,
                              max_iterations, not mask));
    }
  }

  if (const auto mask = rho_max < rest_mass_density_times_lorentz_factor;
      xsimd::any(mask)) {
    // adjust upper bound
    const auto corner_case_function = CornerCaseFunction{
        rest_mass_density_times_lorentz_factor * one_over_rho_min_, r_squared_,
        b_squared_, r_dot_b_squared_};
    using std::min;
    if constexpr (std::is_fundamental_v<T>) {
      upper_bound = min(
          upper_bound, RootFinder::toms748(corner_case_function, lower_bound,
                                           upper_bound, absolute_tolerance,
                                           relative_tolerance, max_iterations));
    } else {
      upper_bound = min(
          upper_bound,
          RootFinder::toms748(corner_case_function, lower_bound, upper_bound,
                              absolute_tolerance, relative_tolerance,
                              max_iterations, not mask));
    }
  }

  return {lower_bound, upper_bound};
}

template <typename T, typename Eos>
Primitives<T> FunctionOfMu<T, Eos>::primitives(const T mu) const {
  using std::min;
  using xsimd::fma;
  using xsimd::fnma;
  // Equation (26)
  const T x = compute_x(mu, b_squared_);
  // Equations(38)
  const T r_bar_squared =
      compute_r_bar_squared(mu, x, r_squared_, r_dot_b_squared_);
  // Equation (40)
  const T mu_squared = square(mu);
  const T v_hat_squared =
      min(mu_squared * r_bar_squared, v_0_squared_);
  // const T w_hat = 1.0 / sqrt(1.0 - v_hat_squared);
  const T one_over_w_hat = sqrt(1.0 - v_hat_squared);
  // Equation (41) with bounds from Equation (5)
  const T rho_hat =
      xsimd::clip(rest_mass_density_times_lorentz_factor_ * one_over_w_hat,
                  T(equation_of_state_.rest_mass_density_lower_bound()),
                  T(equation_of_state_.rest_mass_density_upper_bound()));
  // Equations (39) and (25)
  const T q_bar = q_ - 0.5 * b_squared_ -
                  0.5 * mu_squared * square(x) *
                      (r_squared_ * b_squared_ - r_dot_b_squared_);
  // Equation (42) with bounds from Equation (6)
  const T epsilon_hat = xsimd::clip(
      fma(fnma(mu, r_bar_squared, q_bar), (one_over_w_hat + 1.0),
          v_hat_squared) /
          fma(one_over_w_hat, one_over_w_hat, one_over_w_hat),

      T(equation_of_state_.specific_internal_energy_lower_bound(rho_hat)),
      T(equation_of_state_.specific_internal_energy_upper_bound(rho_hat)));
  // Pressure from EOS
  T p_hat = std::numeric_limits<T>::signaling_NaN();
  if constexpr (Eos::thermodynamic_dim == 1) {
    p_hat =
        get(equation_of_state_.pressure_from_density(Scalar<T>(rho_hat)));
  } else if constexpr (Eos::thermodynamic_dim == 2) {
    p_hat = get(equation_of_state_.pressure_from_density_and_energy(
        Scalar<T>(rho_hat), Scalar<T>(epsilon_hat)));
  } else if constexpr (Eos::thermodynamic_dim == 3) {
    ERROR("3d EOS not implemented");
  }
  return {rho_hat, one_over_w_hat, p_hat, epsilon_hat, q_bar, r_bar_squared};
}

template <typename T, typename Eos>
T FunctionOfMu<T, Eos>::operator()(const T mu) const {
  const auto [rho_hat, one_over_w_hat, p_hat, epsilon_hat, q_bar,
              r_bar_squared] = primitives(mu);
  // Original implementation from Kastaun et al.
  //
  // // Equation (43)
  // const T a_hat = p_hat / (rho_hat * (1.0 + epsilon_hat));
  // const T h_hat = (1.0 + epsilon_hat) * (1.0 + a_hat);
  // // Equations (46) - (48)
  // using std::max;
  // const T nu_hat = max(h_hat * one_over_w_hat,
  //                      (1.0 + a_hat) * (1.0 + q_bar - mu * r_bar_squared));
  // // Equations (44) - (45)
  // return mu - 1.0 / (nu_hat + mu * r_bar_squared);

  // Optimized implementation to minimize FLOPs and specifically divides
  //
  // Equations (44) - (48)
  using std::max;
  const T nu_hat = max((1.0 + epsilon_hat) * one_over_w_hat,
                       xsimd::fnma(mu, r_bar_squared, 1.0 + q_bar));
  const T a_hat_denom = rho_hat * (1.0 + epsilon_hat);
  return mu -
         a_hat_denom /
             xsimd::fma(nu_hat, p_hat,
                        a_hat_denom * xsimd::fma(mu, r_bar_squared, nu_hat));
}
}  // namespace

struct GetFromXsimd {
  template <typename T>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(T& t, const size_t i) const {
    return t.get(i);
  }
};

template <size_t ThermodynamicDim, typename T>
std::optional<PrimitiveRecoveryData> KastaunEtAl::apply(
    const T /*initial_guess_pressure*/, const T total_energy_density,
    const T momentum_density_squared,
    const T momentum_density_dot_magnetic_field, const T magnetic_field_squared,
    const T rest_mass_density_times_lorentz_factor, const T electron_fraction,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  // Master function see Equation (44)
  const auto f_of_mu = FunctionOfMu<T, EquationsOfState::IdealFluid<true>>{
      total_energy_density,
      momentum_density_squared,
      momentum_density_dot_magnetic_field,
      magnetic_field_squared,
      rest_mass_density_times_lorentz_factor,
      electron_fraction,
      dynamic_cast<const EquationsOfState::IdealFluid<true>&>(
          equation_of_state)};

  // mu is 1 / (h W) see Equation (26)
  T one_over_specific_enthalpy_times_lorentz_factor(
      std::numeric_limits<double>::signaling_NaN());
  try {
    // Bracket for master function, see Sec. II.F
    const auto[lower_bound, upper_bound] = f_of_mu.root_bracket(
        rest_mass_density_times_lorentz_factor, absolute_tolerance_,
        relative_tolerance_, max_iterations_);
    // Try to recover primitves
    one_over_specific_enthalpy_times_lorentz_factor =
        // NOLINTNEXTLINE(clang-analyzer-core)
        RootFinder::toms748(f_of_mu, lower_bound, upper_bound,
                            absolute_tolerance_, relative_tolerance_,
                            max_iterations_);

    return PrimitiveRecoveryData{get_element(lower_bound, 0, GetFromXsimd{}),
                                 get_element(upper_bound, 0, GetFromXsimd{}),
                                 0., 0., 0.};

  } catch (std::exception& exception) {
    return std::nullopt;
  }

  const auto[rest_mass_density, one_over_lorentz_factor, pressure,
             specific_internal_energy, q_bar, r_bar_squared] =
      f_of_mu.primitives(one_over_specific_enthalpy_times_lorentz_factor);

  (void)(specific_internal_energy);
  (void)(q_bar);
  (void)(r_bar_squared);
  const T lorentz_factor = 1.0 / one_over_lorentz_factor;
  // TODO: not rho, some other thing.
  const T rho = rest_mass_density_times_lorentz_factor /
                one_over_specific_enthalpy_times_lorentz_factor;

  // TODO: return correct thing.
  return PrimitiveRecoveryData{
    get_element(rest_mass_density, 0, GetFromXsimd{}),
      get_element(lorentz_factor, 0, GetFromXsimd{}),
      get_element(pressure, 0, GetFromXsimd{}),
      get_element(rho, 0, GetFromXsimd{}),
      get_element(electron_fraction, 0, GetFromXsimd{})};
}
}  // namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                               \
  template std::optional<grmhd::ValenciaDivClean::PrimitiveRecoverySchemes:: \
                             PrimitiveRecoveryData>                          \
  grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl::apply<     \
      THERMODIM(data), DTYPE(data)>(                                         \
      const DTYPE(data) initial_guess_pressure,                              \
      const DTYPE(data) total_energy_density,                                \
      const DTYPE(data) momentum_density_squared,                            \
      const DTYPE(data) momentum_density_dot_magnetic_field,                 \
      const DTYPE(data) magnetic_field_squared,                              \
      const DTYPE(data) rest_mass_density_times_lorentz_factor,              \
      const DTYPE(data) electron_fraction,                                   \
      const EquationsOfState::EquationOfState<true, THERMODIM(data)>&        \
          equation_of_state);

namespace {
template <size_t N>
using helper = typename ::xsimd::make_sized_batch<double, N>::type;
}  // namespace

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2),
                        (double, helper<8>, helper<4>, helper<2>))

#undef INSTANTIATION
#undef THERMODIM
