#include <iostream>

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <memory>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/ExtractPoint.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Simd/Simd.hpp"
#include "Utilities/TMPL.hpp"

#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

namespace {
using SimdType = typename simd::make_sized_batch<double, 8>::type;

// This class codes Eq. (B.34), rewritten as a standard-form
// polynomial in (W - 1) or (W - ((tau/D) - (B^2/D) + 1)) for better
// numerical behavior.
//
// If lower_bound is 0, the implemented function is
// ((B^2/D) / 2 - (tau/D)) (1 + 2 (B^2/D) mu^2 + (B^2/D)^2 mu^2)
// + (W-1) (2 ((B^2/D) - (tau/D)) (1 + (B^2/D) mu^2) + (B^2/D) mu^2 + 1)
// + (W-1)^2 ((B^2/D) - (tau/D) + 3/2 (B^2/D) mu^2 + 2)
// + (W-1)^3
// where mu^2 = (B.S)^2 / (B^2 S^2).  A nice property of this form is
// that, because we've already modified tau to satisfy B.39, its value
// at W-1 = 0 is guaranteed to be negative, even in the presence of
// roundoff error.
//
// If lower_bound is LB = ((tau/D) - (B^2/D)), the implemented function is
// - 1/2 (B^2/D) (1 + mu^2 (tau/D) ((tau/D) + 2))
// + (W-[LB+1]) (1 + (B^2/D) mu^2 + LB ((B^2/D) mu^2 + LB + 2))
// + (W-[LB+1])^2 (2 LB + 3/2 (B^2/D) mu^2 + 2)
// + (W-[LB+1])^3
// This form is only used if LB > 0, so, as in the previous case, it
// is guaranteed to be negative at W-[LB+1] = 0.
template <typename T>
class FunctionOfLorentzFactor {
 public:
  FunctionOfLorentzFactor(const T& b_squared_over_d, const T& tau_over_d,
                          const T& normalized_s_dot_b, const T& lower_bound) {
    if constexpr (std::is_fundamental_v<T>) {
      coefficients_ =
          (lower_bound == 0.0
               ? zero_bound(b_squared_over_d, tau_over_d, normalized_s_dot_b)
               : nonzero_bound(b_squared_over_d, tau_over_d, normalized_s_dot_b,
                               lower_bound));
    } else {
#ifdef SPECTRE_USE_SIMD
      const auto mask = lower_bound == 0.0;
      const auto zero_bound_values =
          zero_bound(b_squared_over_d, tau_over_d, normalized_s_dot_b);
      const auto nonzero_bound_values = nonzero_bound(
          b_squared_over_d, tau_over_d, normalized_s_dot_b, lower_bound);
      for (size_t i = 0; i < 4; ++i) {
        gsl::at(coefficients_, i) =
            simd::select(mask, gsl::at(zero_bound_values, i),
                         gsl::at(nonzero_bound_values, i));
      }
#else   // SPECTRE_USE_SIMD
      ERROR("Expected SIMD support but didn't get it.");
#endif  // SPECTRE_USE_SIMD
    }
  }

  T operator()(const T excess_lorentz_factor) const {
    return evaluate_polynomial(coefficients_, excess_lorentz_factor);
  }

 private:
  static std::array<T, 4> zero_bound(const T& b_squared_over_d,
                                     const T& tau_over_d,
                                     const T& normalized_s_dot_b) {
    return std::array{
        (0.5 * b_squared_over_d - tau_over_d) *
            (square(normalized_s_dot_b) * b_squared_over_d *
                 (b_squared_over_d + 2.0) +
             1.0),
        2.0 * (square(normalized_s_dot_b) * b_squared_over_d + 1.0) *
                (b_squared_over_d - tau_over_d) +
            b_squared_over_d * square(normalized_s_dot_b) + 1.0,
        b_squared_over_d - tau_over_d +
            1.5 * square(normalized_s_dot_b) * b_squared_over_d + 2.0,
        T(1.0)};
  }

  static std::array<T, 4> nonzero_bound(const T& b_squared_over_d,
                                        const T& tau_over_d,
                                        const T& normalized_s_dot_b,
                                        const T& lower_bound) {
    return std::array{
        -0.5 * b_squared_over_d *
            (1.0 +
             square(normalized_s_dot_b) * tau_over_d * (tau_over_d + 2.0)),
        1.0 + square(normalized_s_dot_b) * b_squared_over_d +
            lower_bound * (2.0 + square(normalized_s_dot_b) * b_squared_over_d +
                           lower_bound),
        2.0 + 1.5 * square(normalized_s_dot_b) * b_squared_over_d +
            2.0 * lower_bound,
        T(1.0)};
  }

  std::array<T, 4> coefficients_{};
};

template <typename T>
FunctionOfLorentzFactor(const T& b_squared_over_d, const T& tau_over_d,
                        const T& normalized_s_dot_b, const T& lower_bound)
    -> FunctionOfLorentzFactor<T>;

#ifdef SPECTRE_USE_SIMD

template <typename T, size_t... Is>
T make_sequence_impl(std::index_sequence<Is...>) {
  return T{static_cast<typename T::value_type>(Is)...};
}

template <typename T>
T make_sequence() {
  return make_sequence_impl<T>(std::make_index_sequence<T::size>{});
}

template <typename SimdType, typename Symm, typename IndexList>
Tensor<SimdType, Symm, IndexList> extract_point(
    const Tensor<DataVector, Symm, IndexList>& tensor, const size_t index) {
  Tensor<SimdType, Symm, IndexList> result{};
  for (size_t storage_index = 0; storage_index < tensor.size();
       ++storage_index) {
    result[storage_index] =
        SimdType::load_unaligned(std::addressof(tensor[storage_index][index]));
  }
  return result;
}

[[nodiscard]] bool fix_impl_simd(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const double minimum_rest_mass_density_times_lorentz_factor,
    const double one_minus_safety_factor_for_magnetic_field,
    const double one_minus_safety_factor_for_momentum_density,
    const double rest_mass_density_times_lorentz_factor_cutoff,
    const double minimum_electron_fraction,
    const double electron_fraction_cutoff,
    const size_t size,  //
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& tilde_b_squared,
    const Scalar<DataVector>& tilde_s_squared,
    const Scalar<DataVector>& tilde_s_dot_tilde_b,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric) {
  bool needed_fixing = false;
  const SimdType min_rho_times_w(
      minimum_rest_mass_density_times_lorentz_factor),
      two_times_one_min_saf_mag_field(
          one_minus_safety_factor_for_magnetic_field * 2.),
      zero(0.);

  ASSERT(size >= SimdType::size, "Too few points for SIMD operations. size: "
                                     << size
                                     << " SimdType::size: " << SimdType::size);

  bool done = false;
  typename SimdType::batch_bool_type incomplete_mask(true);
  for (size_t s = 0; s < size and not done; s += SimdType::size) {
    if (UNLIKELY(s + SimdType::size > size)) {
      done = true;
      const size_t remainder = s - (size - SimdType::size);
      incomplete_mask = make_sequence<SimdType>() >=
                        static_cast<double>(remainder);

      // Use remainder to offset index
      s = size - SimdType::size;
    }
    auto d_tilde = SimdType::load_unaligned(&get(*tilde_d)[s]);

    // Increase electron fraction if necessary
    auto ye_tilde = SimdType::load_unaligned(&get(*tilde_ye)[s]);
    if (const auto ye_mask =
            ye_tilde < electron_fraction_cutoff * d_tilde and incomplete_mask;
        UNLIKELY(simd::any(ye_mask))) {
      needed_fixing = true;
      ye_tilde =
          simd::select(ye_mask, minimum_electron_fraction * d_tilde, ye_tilde);
      ye_tilde.store_unaligned(&get(*tilde_ye)[s]);
    }

    // Increase density if necessary
    const auto sqrt_det_gamma =
        SimdType::load_unaligned(&get(sqrt_det_spatial_metric)[s]);
    if (const auto d_tilde_mask =
            (d_tilde <
             rest_mass_density_times_lorentz_factor_cutoff * sqrt_det_gamma) and
            incomplete_mask;
        UNLIKELY(simd::any(d_tilde_mask))) {
      needed_fixing = true;
      ye_tilde = ye_tilde / d_tilde;
      d_tilde =
          simd::select(d_tilde_mask, min_rho_times_w * sqrt_det_gamma, d_tilde);
      d_tilde.store_unaligned(&get(*tilde_d)[s]);
      ye_tilde = ye_tilde * d_tilde;
      ye_tilde.store_unaligned(&get(*tilde_ye)[s]);
    }

    // Increase internal energy if necessary
    auto tau_tilde = SimdType::load_unaligned(&get(*tilde_tau)[s]);
    const auto b_tilde_squared =
        SimdType::load_unaligned(&get(tilde_b_squared)[s]);
    // Equation B.39 of Foucart
    const auto mag_temp = two_times_one_min_saf_mag_field * sqrt_det_gamma;
    const auto tau_tilde_mask =
        (b_tilde_squared > (tau_tilde * mag_temp)) and incomplete_mask;
    if (UNLIKELY(simd::any(tau_tilde_mask))) {
      needed_fixing = true;
      tau_tilde =
          simd::select(tau_tilde_mask, b_tilde_squared / mag_temp, tau_tilde);
      tau_tilde.store_unaligned(&get(*tilde_tau)[s]);
    }

    // Decrease momentum density if necessary
    const auto s_tilde_squared =
        SimdType::load_unaligned(&get(tilde_s_squared)[s]);
    // Equation B.24 of Foucart
    const auto tau_over_d = tau_tilde / d_tilde;
    // Equation B.23 of Foucart
    const auto b_squared_over_d = b_tilde_squared / (sqrt_det_gamma * d_tilde);
    // Equation B.27 of Foucart
    const auto normalized_s_dot_b =
        simd::select((b_tilde_squared > 1.e-16 * d_tilde and
                      s_tilde_squared > 1.e-16 * square(d_tilde)),
                     SimdType::load_unaligned(&get(tilde_s_dot_tilde_b)[s]) /
                         sqrt(b_tilde_squared * s_tilde_squared),
                     zero);

    // Equation B.40 of Foucart
    const auto lower_bound_of_lorentz_factor_minus_one =
        simd::max(tau_over_d - b_squared_over_d, zero);
    // Equation B.31 of Foucart
    const auto upper_bound_for_s_tilde_squared =
        [&b_squared_over_d, &d_tilde, &lower_bound_of_lorentz_factor_minus_one,
         &normalized_s_dot_b](
            const SimdType& local_excess_lorentz_factor) {
          const auto local_lorentz_factor_minus_one =
              lower_bound_of_lorentz_factor_minus_one +
              local_excess_lorentz_factor;
          return square(1.0 + local_lorentz_factor_minus_one +
                        b_squared_over_d) *
                 local_lorentz_factor_minus_one *
                 (2.0 + local_lorentz_factor_minus_one) /
                 (square(1.0 + local_lorentz_factor_minus_one) +
                  square(normalized_s_dot_b) * b_squared_over_d *
                      (b_squared_over_d +
                       2.0 * (1.0 + local_lorentz_factor_minus_one))) *
                 square(d_tilde);
        };
    const auto simple_upper_bound_for_s_tilde_squared =
        upper_bound_for_s_tilde_squared(0.0);

    // If s_tilde_squared is small enough, no fix is needed. Otherwise, we need
    // to do some real work.
    if (const auto tilde_s_mask =
            (s_tilde_squared > one_minus_safety_factor_for_momentum_density *
                                   simple_upper_bound_for_s_tilde_squared) and
            incomplete_mask;
        UNLIKELY(simd::any(tilde_s_mask))) {
      // Find root of Equation B.34 of Foucart
      // NOTE:
      // - This assumes minimum specific enthalpy is 1.
      // - SpEC implements a more complicated formula (B.32) which is equivalent
      // - Bounds on root are given by Equation  B.40 of Foucart
      // - In regions where the solution is just above atmosphere we sometimes
      //   obtain an upper bound on the Lorentz factor somewhere around ~1e5,
      //   while the actual Lorentz factor is only 1+1e-6. This leads to
      //   situations where the solver must perform many (over 50) iterations to
      //   converge. A simple way of avoiding this is to check that
      //   [W_{lower_bound}, 10 * W_{lower_bound}] brackets the root and then
      //   use 10 * W_{lower_bound} as the upper bound. This reduces the number
      //   of iterations for the TOMS748 algorithm to converge to less than 10.
      //   Note that the factor 10 is chosen arbitrarily and could probably be
      //   reduced if required. The reasoning behind 10 is that it is unlikely
      //   the Lorentz factor will increase by a factor of 10 from one time step
      //   to the next in a physically meaning situation, and so 10 provides a
      //   reasonable bound.
      const auto f_of_lorentz_factor = FunctionOfLorentzFactor{
          b_squared_over_d, tau_over_d, normalized_s_dot_b,
          lower_bound_of_lorentz_factor_minus_one};
      auto upper_bound =
          simd::select(lower_bound_of_lorentz_factor_minus_one == 0.0,
                       tau_over_d, b_squared_over_d);

      SimdType excess_lorentz_factor(0.0);
      if (simd::any(upper_bound != 0.0)) {
        const SimdType f_at_lower = f_of_lorentz_factor(0.0);
        const SimdType candidate_upper_bound =
            9.0 * (lower_bound_of_lorentz_factor_minus_one + 1.0);
        SimdType f_at_upper(
            std::numeric_limits<double>::signaling_NaN());
        // TODO: double check logic...
        const auto upper_bound_less_candidate_mask =
            upper_bound < candidate_upper_bound;
        if (UNLIKELY(simd::all(upper_bound_less_candidate_mask))) {
          f_at_upper = f_of_lorentz_factor(upper_bound);
        } else {
          // Use f_at_upper_bound to eliminate recalculating it
          const auto f_at_upper_bound = f_of_lorentz_factor(upper_bound);
          f_at_upper =
              simd::select(upper_bound_less_candidate_mask, f_at_upper_bound,
                           f_of_lorentz_factor(candidate_upper_bound));

          const auto f_positive_mask = f_at_upper > 0.0;
          upper_bound = simd::select(
              f_positive_mask and (not upper_bound_less_candidate_mask),
              candidate_upper_bound, upper_bound);
          f_at_upper = simd::select(
              f_at_upper <= 0.0 and (not upper_bound_less_candidate_mask),
              f_at_upper_bound, f_at_upper);
        }
        // Implemented above, but double check correctness.
        // if (upper_bound < candidate_upper_bound) {
        //   f_at_upper = f_of_lorentz_factor(upper_bound);
        // } else {
        //   f_at_upper = f_of_lorentz_factor(candidate_upper_bound);
        //   if (f_at_upper > 0.0) {
        //     upper_bound = candidate_upper_bound;
        //   } else {
        //     f_at_upper = f_of_lorentz_factor(upper_bound);
        //   }
        // }

        try {
          excess_lorentz_factor = RootFinder::toms748(
              f_of_lorentz_factor, SimdType(0.0), upper_bound, f_at_lower,
              f_at_upper, 1.e-14, 1.e-14, 100, not tilde_s_mask);
        } catch (std::exception& exception) {
          // clang-format makes the streamed text hard to read in code...
          // clang-format off
        ERROR(
            "Failed to fix conserved variables because the root finder failed"
            "to find the lorentz factor.\n"
            "upper_bound = "
            << std::scientific << std::setprecision(18)
            << upper_bound
            << "\n  lower_bound_of_lorentz_factor_minus_one = "
            << lower_bound_of_lorentz_factor_minus_one
            << "\n  s_tilde_squared = " << s_tilde_squared
            << "\n  d_tilde = " << d_tilde
            << "\n  sqrt_det_g = " << sqrt_det_gamma
            << "\n  tau_tilde = " << tau_tilde
            << "\n  b_tilde_squared = " << b_tilde_squared
            << "\n  b_squared_over_d = " << b_squared_over_d
            << "\n  tau_over_d = " << tau_over_d
            << "\n  normalized_s_dot_b = " << normalized_s_dot_b
            << "\n  tilde_s =\n" << extract_point<SimdType>(*tilde_s, s)
            << "\n  tilde_b =\n" << extract_point<SimdType>(tilde_b, s)
            << "\n  spatial_metric =\n"
            << extract_point<SimdType>(spatial_metric, s)
            << "\n  inv_spatial_metric =\n"
            << extract_point<SimdType>(inv_spatial_metric, s) << "\n"
            << "The message of the exception thrown by the root finder "
               "is:\n"
            << exception.what());
          // clang-format on
        }
      }  // if (simd::any(upper_bound != 0.0))

      const auto rescaling_factor = simd::min(
          simd::select(
              tilde_s_mask,
              sqrt(one_minus_safety_factor_for_momentum_density *
                   upper_bound_for_s_tilde_squared(excess_lorentz_factor) /
                   (s_tilde_squared + 1.e-16 * square(d_tilde))),
              SimdType(1.)),
          SimdType(1.));
      if (UNLIKELY(simd::any(rescaling_factor < 1.))) {
        needed_fixing = true;
        for (size_t i = 0; i < 3; i++) {
          auto s_tilde = SimdType::load_unaligned(&tilde_s->get(i)[s]);
          s_tilde *= rescaling_factor;
          s_tilde.store_unaligned(&tilde_s->get(i)[s]);
        }
      }
    }
  }

  return needed_fixing;
}
#endif  // SPECTRE_USE_SIMD

[[nodiscard]] bool fix_impl_scalar(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const double minimum_rest_mass_density_times_lorentz_factor,
    const double one_minus_safety_factor_for_magnetic_field,
    const double one_minus_safety_factor_for_momentum_density,
    const double rest_mass_density_times_lorentz_factor_cutoff,
    const double minimum_electron_fraction,
    const double electron_fraction_cutoff,
    const size_t size,  //
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& tilde_b_squared,
    const Scalar<DataVector>& tilde_s_squared,
    const Scalar<DataVector>& tilde_s_dot_tilde_b,
    // Below used for printing only
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric) {
  bool needed_fixing = false;
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>>>
      temp_buffer(size);

  for (size_t s = 0; s < size; s++) {
    double& d_tilde = get(*tilde_d)[s];

    // Increase electron fraction if necessary
    double& ye_tilde = get(*tilde_ye)[s];
    if (ye_tilde < electron_fraction_cutoff * d_tilde) {
      needed_fixing = true;
      ye_tilde = minimum_electron_fraction * d_tilde;
    }

    // Increase mass density if necessary
    const double sqrt_det_g = get(sqrt_det_spatial_metric)[s];
    if (d_tilde < rest_mass_density_times_lorentz_factor_cutoff * sqrt_det_g) {
      needed_fixing = true;
      ye_tilde = ye_tilde / d_tilde;
      d_tilde = minimum_rest_mass_density_times_lorentz_factor * sqrt_det_g;
      ye_tilde = ye_tilde * d_tilde;
    }

    // Increase internal energy if necessary
    double& tau_tilde = get(*tilde_tau)[s];
    const double b_tilde_squared = get(tilde_b_squared)[s];
    // Equation B.39 of Foucart
    if (b_tilde_squared > one_minus_safety_factor_for_magnetic_field * 2. *
                              tau_tilde * sqrt_det_g) {
      needed_fixing = true;
      tau_tilde = 0.5 * b_tilde_squared /
                  (one_minus_safety_factor_for_magnetic_field * sqrt_det_g);
    }
  }

  for (size_t s = 0; s < size; s++) {
    const double d_tilde = get(*tilde_d)[s];
    const double sqrt_det_g = get(sqrt_det_spatial_metric)[s];
    const double tau_tilde = get(*tilde_tau)[s];
    const double b_tilde_squared = get(tilde_b_squared)[s];
    // Decrease momentum density if necessary
    const double s_tilde_squared = get(tilde_s_squared)[s];
    // Equation B.24 of Foucart
    const double tau_over_d = tau_tilde / d_tilde;
    // Equation B.23 of Foucart
    const double b_squared_over_d = b_tilde_squared / sqrt_det_g / d_tilde;
    // Equation B.27 of Foucart
    const double normalized_s_dot_b =
        (b_tilde_squared > 1.e-16 * d_tilde and
         s_tilde_squared > 1.e-16 * square(d_tilde))
            ? get(tilde_s_dot_tilde_b)[s] /
                  sqrt(b_tilde_squared * s_tilde_squared)
            : 0.;

    // Equation B.40 of Foucart
    const double lower_bound_of_lorentz_factor_minus_one =
        std::max(tau_over_d - b_squared_over_d, 0.);
    // Equation B.31 of Foucart
    const auto upper_bound_for_s_tilde_squared =
        [&b_squared_over_d, &d_tilde, &lower_bound_of_lorentz_factor_minus_one,
         &normalized_s_dot_b](const double local_excess_lorentz_factor) {
          const double local_lorentz_factor_minus_one =
              lower_bound_of_lorentz_factor_minus_one +
              local_excess_lorentz_factor;
          return square(1.0 + local_lorentz_factor_minus_one +
                        b_squared_over_d) *
                 local_lorentz_factor_minus_one *
                 (2.0 + local_lorentz_factor_minus_one) /
                 (square(1.0 + local_lorentz_factor_minus_one) +
                  square(normalized_s_dot_b) * b_squared_over_d *
                      (b_squared_over_d +
                       2.0 * (1.0 + local_lorentz_factor_minus_one))) *
                 square(d_tilde);
        };
    const double simple_upper_bound_for_s_tilde_squared =
        upper_bound_for_s_tilde_squared(0.0);

    // If s_tilde_squared is small enough, no fix is needed. Otherwise, we need
    // to do some real work.
    if (s_tilde_squared > one_minus_safety_factor_for_momentum_density *
                              simple_upper_bound_for_s_tilde_squared) {
      // Find root of Equation B.34 of Foucart
      // NOTE:
      // - This assumes minimum specific enthalpy is 1.
      // - SpEC implements a more complicated formula (B.32) which is equivalent
      // - Bounds on root are given by Equation  B.40 of Foucart
      // - In regions where the solution is just above atmosphere we sometimes
      //   obtain an upper bound on the Lorentz factor somewhere around ~1e5,
      //   while the actual Lorentz factor is only 1+1e-6. This leads to
      //   situations where the solver must perform many (over 50) iterations to
      //   converge. A simple way of avoiding this is to check that
      //   [W_{lower_bound}, 10 * W_{lower_bound}] brackets the root and then
      //   use 10 * W_{lower_bound} as the upper bound. This reduces the number
      //   of iterations for the TOMS748 algorithm to converge to less than 10.
      //   Note that the factor 10 is chosen arbitrarily and could probably be
      //   reduced if required. The reasoning behind 10 is that it is unlikely
      //   the Lorentz factor will increase by a factor of 10 from one time step
      //   to the next in a physically meaning situation, and so 10 provides a
      //   reasonable bound.
      const auto f_of_lorentz_factor = FunctionOfLorentzFactor{
          b_squared_over_d, tau_over_d, normalized_s_dot_b,
          lower_bound_of_lorentz_factor_minus_one};
      double upper_bound = lower_bound_of_lorentz_factor_minus_one == 0.0
                               ? tau_over_d
                               : b_squared_over_d;

      double excess_lorentz_factor = 0.0;
      if (upper_bound != 0.0) {
        const double f_at_lower = f_of_lorentz_factor(0.0);
        const double candidate_upper_bound =
            9.0 * (lower_bound_of_lorentz_factor_minus_one + 1.0);
        double f_at_upper = std::numeric_limits<double>::signaling_NaN();
        if (upper_bound < candidate_upper_bound) {
          f_at_upper = f_of_lorentz_factor(upper_bound);
        } else {
          f_at_upper = f_of_lorentz_factor(candidate_upper_bound);
          if (f_at_upper > 0.0) {
            upper_bound = candidate_upper_bound;
          } else {
            f_at_upper = f_of_lorentz_factor(upper_bound);
          }
        }

        try {
          excess_lorentz_factor =
              RootFinder::toms748(f_of_lorentz_factor, 0.0, upper_bound,
                                  f_at_lower, f_at_upper, 1.e-14, 1.e-14, 100);
        } catch (std::exception& exception) {
          // clang-format makes the streamed text hard to read in code...
          // clang-format off
        ERROR(
            "Failed to fix conserved variables because the root finder failed "
            "to find the lorentz factor.\n"
            "  upper_bound = "
            << std::scientific << std::setprecision(18)
            << upper_bound
            << "\n  lower_bound_of_lorentz_factor_minus_one = "
            << lower_bound_of_lorentz_factor_minus_one
            << "\n  s_tilde_squared = " << s_tilde_squared
            << "\n  d_tilde = " << d_tilde
            << "\n  sqrt_det_g = " << sqrt_det_g
            << "\n  tau_tilde = " << tau_tilde
            << "\n  b_tilde_squared = " << b_tilde_squared
            << "\n  b_squared_over_d = " << b_squared_over_d
            << "\n  tau_over_d = " << tau_over_d
            << "\n  normalized_s_dot_b = " << normalized_s_dot_b
            << "\n  tilde_s =\n" << extract_point(*tilde_s, s)
            << "\n  tilde_b =\n" << extract_point(tilde_b, s)
            << "\n  spatial_metric =\n" << extract_point(spatial_metric, s)
            << "\n  inv_spatial_metric =\n"
            << extract_point(inv_spatial_metric, s) << "\n"
            << "The message of the exception thrown by the root finder "
               "is:\n"
            << exception.what());
          // clang-format on
        }
      }

      const double rescaling_factor =
          sqrt(one_minus_safety_factor_for_momentum_density *
               upper_bound_for_s_tilde_squared(excess_lorentz_factor) /
               (s_tilde_squared + 1.e-16 * square(d_tilde)));
      if (rescaling_factor < 1.) {
        needed_fixing = true;
        for (size_t i = 0; i < 3; i++) {
          tilde_s->get(i)[s] *= rescaling_factor;
        }
      }
    }
  }
  return needed_fixing;
}
}  // namespace

namespace grmhd::ValenciaDivClean {
FixConservatives::FixConservatives(
    const double minimum_rest_mass_density_times_lorentz_factor,
    const double rest_mass_density_times_lorentz_factor_cutoff,
    const double minimum_electron_fraction,
    const double electron_fraction_cutoff,
    const double safety_factor_for_magnetic_field,
    const double safety_factor_for_momentum_density,
    const Options::Context& context)
    : minimum_rest_mass_density_times_lorentz_factor_(
          minimum_rest_mass_density_times_lorentz_factor),
      rest_mass_density_times_lorentz_factor_cutoff_(
          rest_mass_density_times_lorentz_factor_cutoff),
      minimum_electron_fraction_(minimum_electron_fraction),
      electron_fraction_cutoff_(electron_fraction_cutoff),
      one_minus_safety_factor_for_magnetic_field_(
          1.0 - safety_factor_for_magnetic_field),
      one_minus_safety_factor_for_momentum_density_(
          1.0 - safety_factor_for_momentum_density) {
  if (minimum_rest_mass_density_times_lorentz_factor_ >
      rest_mass_density_times_lorentz_factor_cutoff_) {
    PARSE_ERROR(context,
                "The minimum value of D (a.k.a. rest mass density times "
                "Lorentz factor) ("
                    << minimum_rest_mass_density_times_lorentz_factor_
                    << ") must be less than or equal to the cutoff value of D ("
                    << rest_mass_density_times_lorentz_factor_cutoff_ << ')');
  }
  if (minimum_electron_fraction_ > electron_fraction_cutoff_) {
    PARSE_ERROR(context,
                "The minimum value of electron fraction Y_e ("
                    << minimum_electron_fraction_
                    << ") must be less than or equal to the cutoff value ("
                    << electron_fraction_cutoff_ << ')');
  }
}

// NOLINTNEXTLINE(google-runtime-references)
void FixConservatives::pup(PUP::er& p) {
  p | minimum_rest_mass_density_times_lorentz_factor_;
  p | rest_mass_density_times_lorentz_factor_cutoff_;
  p | one_minus_safety_factor_for_magnetic_field_;
  p | one_minus_safety_factor_for_momentum_density_;
}

// WARNING!
// Notation of Foucart is not that of SpECTRE
// SpECTRE           Foucart
// {\tilde D}        \rho_*
// {\tilde \tau}     \tau
// {\tilde S}_k      S_k
// {\tilde B}^k      B^k \sqrt{g}
// \rho              \rho_0
// \gamma_{mn}       g_{mn}
bool FixConservatives::operator()(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) const {
  bool needed_fixing = false;
  const size_t size = get<0>(tilde_b).size();
  Variables<tmpl::list<::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
                       ::Tags::TempScalar<3>>>
      temp_buffer(size);

  Scalar<DataVector>& tilde_b_squared = get<::Tags::TempScalar<1>>(temp_buffer);
  dot_product(make_not_null(&tilde_b_squared), tilde_b, tilde_b,
              spatial_metric);

  Scalar<DataVector>& tilde_s_squared = get<::Tags::TempScalar<2>>(temp_buffer);
  dot_product(make_not_null(&tilde_s_squared), *tilde_s, *tilde_s,
              inv_spatial_metric);

  Scalar<DataVector>& tilde_s_dot_tilde_b =
      get<::Tags::TempScalar<3>>(temp_buffer);
  dot_product(make_not_null(&tilde_s_dot_tilde_b), *tilde_s, tilde_b);

#ifdef SPECTRE_USE_SIMD
  if (size >= SimdType::size
      // and false
  ) {
    needed_fixing = fix_impl_simd(
        tilde_d, tilde_ye, tilde_tau, tilde_s,
        minimum_rest_mass_density_times_lorentz_factor_,
        one_minus_safety_factor_for_magnetic_field_,
        one_minus_safety_factor_for_momentum_density_,
        rest_mass_density_times_lorentz_factor_cutoff_,
        minimum_electron_fraction_, electron_fraction_cutoff_, size,
        sqrt_det_spatial_metric, tilde_b_squared, tilde_s_squared,
        tilde_s_dot_tilde_b, tilde_b, spatial_metric, inv_spatial_metric);
  } else {
    needed_fixing = fix_impl_scalar(
        tilde_d, tilde_ye, tilde_tau, tilde_s,
        minimum_rest_mass_density_times_lorentz_factor_,
        one_minus_safety_factor_for_magnetic_field_,
        one_minus_safety_factor_for_momentum_density_,
        rest_mass_density_times_lorentz_factor_cutoff_,
        minimum_electron_fraction_, electron_fraction_cutoff_, size,
        sqrt_det_spatial_metric, tilde_b_squared, tilde_s_squared,
        tilde_s_dot_tilde_b, tilde_b, spatial_metric, inv_spatial_metric);
  }
#else
  needed_fixing = fix_impl_scalar(
      tilde_d, tilde_tau, tilde_s,
      minimum_rest_mass_density_times_lorentz_factor_,
      one_minus_safety_factor_for_magnetic_field_,
      one_minus_safety_factor_for_momentum_density_,
      rest_mass_density_times_lorentz_factor_cutoff_, size,
      sqrt_det_spatial_metric, rest_mass_density_times_lorentz_factor,
      tilde_b_squared, tilde_s_squared, tilde_s_dot_tilde_b, tilde_b,
      spatial_metric, inv_spatial_metric);
#endif
  return needed_fixing;
}

bool operator==(const FixConservatives& lhs, const FixConservatives& rhs) {
  return lhs.minimum_rest_mass_density_times_lorentz_factor_ ==
             rhs.minimum_rest_mass_density_times_lorentz_factor_ and
         lhs.rest_mass_density_times_lorentz_factor_cutoff_ ==
             rhs.rest_mass_density_times_lorentz_factor_cutoff_ and
         lhs.one_minus_safety_factor_for_magnetic_field_ ==
             rhs.one_minus_safety_factor_for_magnetic_field_ and
         lhs.one_minus_safety_factor_for_momentum_density_ ==
             rhs.one_minus_safety_factor_for_momentum_density_;
}

bool operator!=(const FixConservatives& lhs, const FixConservatives& rhs) {
  return not(lhs == rhs);
}
}  // namespace grmhd::ValenciaDivClean
