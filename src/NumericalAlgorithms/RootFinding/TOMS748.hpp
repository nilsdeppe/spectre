// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function RootFinder::toms748

#pragma once

#include <boost/math/tools/roots.hpp>
#include <functional>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/Simd/Simd.hpp"

namespace RootFinder {
#ifdef SPECTRE_USE_SIMD
namespace toms748_detail {
template <typename T, typename Arch>
simd::batch<T, Arch> secant_interpolate(const simd::batch<T, Arch>& a,
                                        const simd::batch<T, Arch>& b,
                                        const simd::batch<T, Arch>& fa,
                                        const simd::batch<T, Arch>& fb);
template <typename T, typename Arch>
simd::batch<T, Arch> quadratic_interpolate(const simd::batch<T, Arch>& a,
                                           const simd::batch<T, Arch>& b,
                                           const simd::batch<T, Arch>& d,
                                           const simd::batch<T, Arch>& fa,
                                           const simd::batch<T, Arch>& fb,
                                           const simd::batch<T, Arch>& fd,
                                           const unsigned count);
template <typename T, typename Arch>
simd::batch<T, Arch> cubic_interpolate(
    const simd::batch<T, Arch>& a, const simd::batch<T, Arch>& b,
    const simd::batch<T, Arch>& d, const simd::batch<T, Arch>& e,
    const simd::batch<T, Arch>& fa, const simd::batch<T, Arch>& fb,
    const simd::batch<T, Arch>& fd, const simd::batch<T, Arch>& fe);

template <typename T, typename Arch>
simd::batch<T, Arch> safe_div(const simd::batch<T, Arch>& num,
                              const simd::batch<T, Arch>& denom,
                              const simd::batch<T, Arch>& r) {
  // return num / denom without overflow, return r if overflow would occur.
  //
  // This also prevents division by zero.
  const auto mask = (fabs(denom) < (static_cast<T>(1))) and
                    (fabs(denom * std::numeric_limits<T>::max()) <= fabs(num));
  return simd::select(
      mask, r,
      num / simd::select(mask, simd::batch<T, Arch>(static_cast<T>(1)), denom));
}

template <typename T, typename Arch>
simd::batch<T, Arch> secant_interpolate(const simd::batch<T, Arch>& a,
                                        const simd::batch<T, Arch>& b,
                                        const simd::batch<T, Arch>& fa,
                                        const simd::batch<T, Arch>& fb) {
  //
  // Performs standard secant interpolation of [a,b] given
  // function evaluations f(a) and f(b).  Performs a bisection
  // if secant interpolation would leave us very close to either
  // a or b.  Rationale: we only call this function when at least
  // one other form of interpolation has already failed, so we know
  // that the function is unlikely to be smooth with a root very
  // close to a or b.
  //
  const simd::batch<T, Arch> tol_batch =
      simd::batch<T, Arch>(std::numeric_limits<T>::epsilon()) *
      simd::batch<T, Arch>(static_cast<T>(5));
  const simd::batch<T, Arch> c = simd::fnma((fa / (fb - fa)), (b - a), a);
  return simd::select((c <= simd::fma(fabs(a), tol_batch, a)) or
                          (c >= simd::fnma(fabs(b), tol_batch, b)),
                      static_cast<T>(0.5) * (a + b), c);
}

template <typename T, typename Arch>
simd::batch<T, Arch> quadratic_interpolate(const simd::batch<T, Arch>& a,
                                           const simd::batch<T, Arch>& b,
                                           const simd::batch<T, Arch>& d,
                                           const simd::batch<T, Arch>& fa,
                                           const simd::batch<T, Arch>& fb,
                                           const simd::batch<T, Arch>& fd,
                                           const unsigned count) {
  // Performs quadratic interpolation to determine the next point,
  // takes count Newton steps to find the location of the
  // quadratic polynomial.
  //
  // Point d must lie outside of the interval [a,b], it is the third
  // best approximation to the root, after a and b.
  //
  // Note: this does not guarantee to find a root
  // inside [a, b], so we fall back to a secant step should
  // the result be out of range.
  //
  // Start by obtaining the coefficients of the quadratic polynomial:
  const simd::batch<T, Arch> B = safe_div(
      fb - fa, b - a, simd::batch<T, Arch>(std::numeric_limits<T>::max()));
  simd::batch<T, Arch> A = safe_div(
      fd - fb, d - b, simd::batch<T, Arch>(std::numeric_limits<T>::max()));
  A = safe_div(A - B, d - a, simd::batch<T, Arch>(static_cast<T>(0)));

  const auto secant_failure_mask = A == static_cast<T>(0);
  simd::batch<T, Arch> result_secant{};
  if (UNLIKELY(simd::any(secant_failure_mask))) {
    // failure to determine coefficients, try a secant step:
    result_secant = secant_interpolate(a, b, fa, fb);
    if (UNLIKELY(simd::all(secant_failure_mask))) {
      return result_secant;
    }
  }

  // Determine the starting point of the Newton steps:
  simd::batch<T, Arch> c = simd::select(A * fa > static_cast<T>(0), a, b);

  // Take the Newton steps:
  const simd::batch<T, Arch> two_A = static_cast<T>(2) * A;
  const simd::batch<T, Arch> half_a_plus_b = 0.5 * (a + b);
  const simd::batch<T, Arch> one_minus_a = static_cast<T>(1) - a;
  const simd::batch<T, Arch> B_minus_A_times_b = B - A * b;
  for (unsigned i = 1; i <= count; ++i) {
    c -= safe_div(simd::fma(simd::fma(A, c, B_minus_A_times_b), c - a, fa),
                  simd::fma(two_A, c - half_a_plus_b, B), one_minus_a + c);
  }
  if (const auto mask = (c <= a) or (c >= b); simd::any(mask)) {
    // Oops, failure, try a secant step:
    c = simd::select(mask, secant_interpolate(a, b, fa, fb), c);
  }
  return simd::select(secant_failure_mask, result_secant, c);
}

template <typename T, typename Arch>
simd::batch<T, Arch> cubic_interpolate(
    const simd::batch<T, Arch>& a, const simd::batch<T, Arch>& b,
    const simd::batch<T, Arch>& d, const simd::batch<T, Arch>& e,
    const simd::batch<T, Arch>& fa, const simd::batch<T, Arch>& fb,
    const simd::batch<T, Arch>& fd, const simd::batch<T, Arch>& fe) {
  // Uses inverse cubic interpolation of f(x) at points
  // [a,b,d,e] to obtain an approximate root of f(x).
  // Points d and e lie outside the interval [a,b]
  // and are the third and forth best approximations
  // to the root that we have found so far.
  //
  // Note: this does not guarantee to find a root
  // inside [a, b], so we fall back to quadratic
  // interpolation in case of an erroneous result.
  //
  // This commented chunk is the original Boost implementation translated into
  // simd. The actual code below is a heavily optimized version.
  //
  // const simd::batch<T, Arch> q11 = (d - e) * fd / (fe - fd);
  // const simd::batch<T, Arch> q21 = (b - d) * fb / (fd - fb);
  // const simd::batch<T, Arch> q31 = (a - b) * fa / (fb - fa);
  // const simd::batch<T, Arch> d21 = (b - d) * fd / (fd - fb);
  // const simd::batch<T, Arch> d31 = (a - b) * fb / (fb - fa);
  //
  // const simd::batch<T, Arch> q22 = (d21 - q11) * fb / (fe - fb);
  // const simd::batch<T, Arch> q32 = (d31 - q21) * fa / (fd - fa);
  // const simd::batch<T, Arch> d32 = (d31 - q21) * fd / (fd - fa);
  // const simd::batch<T, Arch> q33 = (d32 - q22) * fa / (fe - fa);
  //
  // simd::batch<T, Arch> c = q31 + q32 + q33 + a;

  // The optimized implementation here is L1-cache bound. That is, we aren't
  // able to completely saturate the FP units because we are waiting on the L1
  // cache. While not ideal, that's okay and just part of the algorithm.
  const simd::batch<T, Arch> denom_fb_fa = fb - fa;
  const simd::batch<T, Arch> denom_fd_fb = fd - fb;
  const simd::batch<T, Arch> denom_fd_fa = fd - fa;
  const simd::batch<T, Arch> denom_fe_fd = fe - fd;
  const simd::batch<T, Arch> denom_fe_fb = fe - fb;
  const simd::batch<T, Arch> denom =
      denom_fe_fb * denom_fe_fd * denom_fd_fa * denom_fd_fb * (fe - fa);

  const simd::batch<T, Arch> fa_by_denom = fa / (denom_fb_fa * denom);

  const simd::batch<T, Arch> d31 = (a - b);
  const simd::batch<T, Arch> q21 = (b - d);
  const simd::batch<T, Arch> q32 =
      simd::fms(denom_fd_fb, d31, denom_fb_fa * q21) * denom_fe_fb *
      denom_fe_fd;
  const simd::batch<T, Arch> q22 =
      simd::fms(denom_fe_fd, q21, (d - e) * denom_fd_fb) * fd * denom_fb_fa *
      denom_fd_fa;

  simd::batch<T, Arch> c = simd::fma(
      fa_by_denom,
      simd::fma(fb, simd::fms(q32, (fe + denom_fd_fa), q22), d31 * denom), a);

  if (const auto mask = (c <= a) or (c >= b); simd::any(mask)) {
    // Out of bounds step, fall back to quadratic interpolation:
    c = simd::select(mask, quadratic_interpolate(a, b, d, fa, fb, fd, 3), c);
  }

  return c;
}

template <typename F, typename T, typename Arch>
void bracket(F f, simd::batch<T, Arch>& a, simd::batch<T, Arch>& b,
             simd::batch<T, Arch> c, simd::batch<T, Arch>& fa,
             simd::batch<T, Arch>& fb, simd::batch<T, Arch>& d,
             simd::batch<T, Arch>& fd) {
  // Given a point c inside the existing enclosing interval
  // [a, b] sets a = c if f(c) == 0, otherwise finds the new
  // enclosing interval: either [a, c] or [c, b] and sets
  // d and fd to the point that has just been removed from
  // the interval.  In other words d is the third best guess
  // to the root.
  const simd::batch<T, Arch> tol_batch =
      simd::batch<T, Arch>(std::numeric_limits<T>::epsilon()) *
      simd::batch<T, Arch>(static_cast<T>(2));

  // If the interval [a,b] is very small, or if c is too close
  // to one end of the interval then we need to adjust the
  // location of c accordingly. This is:
  //
  //   if ((b - a) < 2 * tol * a) {
  //     c = a + (b - a) / 2;
  //   } else if (c <= a + fabs(a) * tol) {
  //     c = a + fabs(a) * tol;
  //   } else if (c >= b - fabs(b) * tol) {
  //     c = b - fabs(b) * tol;
  //   }
  const simd::batch<T, Arch> a_filt = simd::fma(fabs(a), tol_batch, a);
  const simd::batch<T, Arch> b_filt = simd::fnma(fabs(b), tol_batch, b);
  const simd::batch<T, Arch> b_minus_a = b - a;
  c = simd::select(
      (b_minus_a < tol_batch * a),
      simd::fma(b_minus_a, simd::batch<T, Arch>(static_cast<T>(0.5)), a),
      simd::select(c <= a_filt, a_filt, simd::select(c >= b_filt, b_filt, c)));

  // Invoke f(c):
  simd::batch<T, Arch> fc = f(c);

  // if we have a zero then we have an exact solution to the root:
  const auto fc_is_zero_mask = fc == static_cast<T>(0);
  if (UNLIKELY(simd::any(fc_is_zero_mask))) {
    a = simd::select(fc_is_zero_mask, c, a);
    fa = simd::select(fc_is_zero_mask, simd::batch<T, Arch>(static_cast<T>(0)),
                      fa);
    d = simd::select(fc_is_zero_mask, simd::batch<T, Arch>(static_cast<T>(0)),
                     d);
    fd = simd::select(fc_is_zero_mask, simd::batch<T, Arch>(static_cast<T>(0)),
                      fd);
    if (UNLIKELY(simd::all(fc_is_zero_mask))) {
      // TODO: add goto restore_completion_mask, or do if (not all()) {...}
      // and have restore completion mask if-else
      return;
    }
  }

  // Non-zero fc, update the interval:
  // Need the != 0 because simd sign is non-zero for zero, but boost::sign is
  // zero at zero. Boost code is:
  // if (boost::math::sign(fa) * boost::math::sign(fc) < 0) {...} else {...}
  const auto sign_mask =
      (fa * fc < static_cast<T>(0)) and (fa != static_cast<T>(0));
  const auto mask_if = sign_mask and (not fc_is_zero_mask);
  d = simd::select(mask_if, b, d);
  fd = simd::select(mask_if, fb, fd);
  b = simd::select(mask_if, c, b);
  fb = simd::select(mask_if, fc, fb);

  const auto mask_else = (not sign_mask) and (not fc_is_zero_mask);
  d = simd::select(mask_else, a, d);
  fd = simd::select(mask_else, fa, fd);
  a = simd::select(mask_else, c, a);
  fa = simd::select(mask_else, fc, fa);
}

template <class F, class T, class Arch, class Tol>
std::pair<simd::batch<T, Arch>, simd::batch<T, Arch>> toms748_solve(
    F f, const simd::batch<T, Arch>& ax, const simd::batch<T, Arch>& bx,
    const simd::batch<T, Arch>& fax, const simd::batch<T, Arch>& fbx, Tol tol,
    size_t& max_iter) {
  static_assert(std::is_floating_point_v<T>);
  // Main entry point and logic for Toms Algorithm 748
  // root finder.
  if (UNLIKELY(simd::any(ax >= bx))) {
    throw std::domain_error("Lower bound is larger than upper bound");
  }

  // Sanity check - are we allowed to iterate at all?
  if (UNLIKELY(max_iter == 0)) {
    return std::pair{ax, bx};
  }

  size_t count = max_iter;
  // mu is a parameter in the algorithm that must be between (0, 1).
  static const T mu = 0.5f;

  // initialise a, b and fa, fb:
  simd::batch<T, Arch> a = ax;
  simd::batch<T, Arch> b = bx;
  simd::batch<T, Arch> fa = fax;
  simd::batch<T, Arch> fb = fbx;

  const auto fa_is_zero_mask = (fa == static_cast<T>(0));
  const auto fb_is_zero_mask = (fb == static_cast<T>(0));
  auto completion_mask = tol(a, b) or fa_is_zero_mask or fb_is_zero_mask;
  if (UNLIKELY(simd::all(completion_mask))) {
    return std::pair{simd::select(fb_is_zero_mask, b, a),
                     simd::select(fa_is_zero_mask, a, b)};
  }

  if (UNLIKELY(simd::any(a * fb > static_cast<T>(0) and
                         (not fa_is_zero_mask) and (not fb_is_zero_mask)))) {
    throw std::domain_error(
        "Parameters lower and upper bounds do not bracket a root");
  }
  // dummy value for fd, e and fe:
  simd::batch<T, Arch> fe(static_cast<T>(1e5F)), e(static_cast<T>(1e5F)),
      fd(static_cast<T>(1e5F));

  simd::batch<T, Arch> c(std::numeric_limits<T>::signaling_NaN()),
      d(std::numeric_limits<T>::signaling_NaN());

  const simd::batch<T, Arch> simd_nan(std::numeric_limits<T>::signaling_NaN());
  auto completed_a = simd::select(completion_mask, a, simd_nan);
  auto completed_b = simd::select(completion_mask, b, simd_nan);
  // TODO: we can eliminate the functions calls on completed_a/b at the end if
  // we also store the completed_fa/fb.
  const auto update_completed = [&fa, &completion_mask, &completed_a,
                                 &completed_b, &a, &b, &tol]() {
    const auto new_completed =
        (fa == static_cast<T>(0) or tol(a, b)) and (not completion_mask);
    completed_a = simd::select(new_completed, a, completed_a);
    completed_b = simd::select(new_completed, b, completed_b);
    completion_mask = new_completed or completion_mask;
    // returns true if _all_ simd registers have been completed
    return simd::all(completion_mask);
  };

  if (simd::any(fa != static_cast<T>(0))) {
    // On the first step we take a secant step:
    c = toms748_detail::secant_interpolate(a, b, fa, fb);
    toms748_detail::bracket(f, a, b, c, fa, fb, d, fd);
    --count;

    if (count and not update_completed()) {
      // On the second step we take a quadratic interpolation:
      c = toms748_detail::quadratic_interpolate(a, b, d, fa, fb, fd, 2);
      e = d;
      fe = fd;
      toms748_detail::bracket(f, a, b, c, fa, fb, d, fd);
      --count;
    }
  }

  simd::batch<T, Arch> u(std::numeric_limits<T>::signaling_NaN()),
      fu(std::numeric_limits<T>::signaling_NaN()),
      a0(std::numeric_limits<T>::signaling_NaN()),
      b0(std::numeric_limits<T>::signaling_NaN());

  // Note: The Boost fa!=0 check is handled with the completion_mask.
  while (count and not simd::all(completion_mask)) {
    // save our brackets:
    a0 = a;
    b0 = b;
    // Starting with the third step taken
    // we can use either quadratic or cubic interpolation.
    // Cubic interpolation requires that all four function values
    // fa, fb, fd, and fe are distinct, should that not be the case
    // then variable prof will get set to true, and we'll end up
    // taking a quadratic step instead.
    static const T min_diff = std::numeric_limits<T>::min() * 32;
    bool prof =
        simd::any(((fabs(fa - fb) < min_diff) or (fabs(fa - fd) < min_diff) or
                   (fabs(fa - fe) < min_diff) or (fabs(fb - fd) < min_diff) or
                   (fabs(fb - fe) < min_diff) or (fabs(fd - fe) < min_diff)) and
                  not completion_mask);
    // TODO: Maybe we should do a cubic _and_ quadratic if we need to? This
    // currently degrades the one find to a quadratic, though realistically I
    // doubt that's a problem.
    //
    // TODO: we need to pass the completion mask down to prevent degrading to
    // lower order computations or adding lower order work when it's not
    // necessary.
    if (prof) {
      c = toms748_detail::quadratic_interpolate(a, b, d, fa, fb, fd, 2);
    } else {
      c = toms748_detail::cubic_interpolate(a, b, d, e, fa, fb, fd, fe);
    }
    // re-bracket, and check for termination:
    e = d;
    fe = fd;
    toms748_detail::bracket(f, a, b, c, fa, fb, d, fd);
    if ((0 == --count) or update_completed()) {
      break;
    }
    // Now another interpolated step:
    prof =
        simd::any(((fabs(fa - fb) < min_diff) or (fabs(fa - fd) < min_diff) or
                   (fabs(fa - fe) < min_diff) or (fabs(fb - fd) < min_diff) or
                   (fabs(fb - fe) < min_diff) or (fabs(fd - fe) < min_diff)) and
                  not completion_mask);
    if (prof) {
      c = toms748_detail::quadratic_interpolate(a, b, d, fa, fb, fd, 3);
    } else {
      c = toms748_detail::cubic_interpolate(a, b, d, e, fa, fb, fd, fe);
    }
    // Bracket again, and check termination condition, update e:
    toms748_detail::bracket(f, a, b, c, fa, fb, d, fd);
    if ((0 == --count) or update_completed()) {
      break;
    }

    // Now we take a double-length secant step:
    const auto fabs_fa_less_fabs_fb_mask = fabs(fa) < fabs(fb);
    u = simd::select(fabs_fa_less_fabs_fb_mask, a, b);
    fu = simd::select(fabs_fa_less_fabs_fb_mask, fa, fb);
    const simd::batch<T, Arch> b_minus_a = b - a;
    // need to mask out fb==fa case
    c = simd::fnma(static_cast<T>(2) * (fu / (fb - fa)), b_minus_a, u);
    // c = u - static_cast<T>(2) * (fu / (fb - fa)) * b_minus_a;
    c = simd::select(
        static_cast<T>(2) * fabs(c - u) > b_minus_a,
        simd::fma(simd::batch<T, Arch>(static_cast<T>(0.5)), b_minus_a, a), c);

    // Bracket again, and check termination condition:
    e = d;
    fe = fd;
    toms748_detail::bracket(f, a, b, c, fa, fb, d, fd);
    if ((0 == --count) or update_completed()) {
      break;
    }

    // And finally... check to see if an additional bisection step is
    // to be taken, we do this if we're not converging fast enough:
    const auto bisection_mask =
        (b - a) >= mu * (b0 - a0) and (not completion_mask);
    if (LIKELY(simd::none(bisection_mask))) {
      continue;
    }
    // bracket again on a bisection:
    const auto e_prebisection = e;
    const auto fe_prebisection = fe;
    const auto a_prebisection = a;
    const auto fa_prebisection = fa;
    const auto b_prebisection = b;
    const auto fb_prebisection = fb;
    const auto d_prebisection = d;
    const auto fd_prebisection = fd;
    e = d;
    fe = fd;
    toms748_detail::bracket(
        f, a, b,
        simd::fma((b - a), simd::batch<T, Arch>(static_cast<T>(0.5)), a), fa,
        fb, d, fd);
    a = simd::select(bisection_mask, a, a_prebisection);
    fa = simd::select(bisection_mask, fa, fa_prebisection);
    b = simd::select(bisection_mask, b, b_prebisection);
    fb = simd::select(bisection_mask, fb, fb_prebisection);
    d = simd::select(bisection_mask, d, d_prebisection);
    fd = simd::select(bisection_mask, fd, fd_prebisection);
    e = simd::select(bisection_mask, e, e_prebisection);
    fe = simd::select(bisection_mask, fe, fe_prebisection);
    --count;
    if (update_completed()) {
      break;
    }
  }  // while loop

  max_iter -= count;
  // TODO: why recompute fa and fb? Does that have to do with completion mask
  fa = f(completed_a);
  fb = f(completed_b);
  completed_b = simd::select(fa == static_cast<T>(0), completed_a, completed_b);
  completed_a = simd::select(fb == static_cast<T>(0), completed_b, completed_a);
  return std::pair{completed_a, completed_b};
}
}  // namespace toms748_detail

template <typename Function, typename T, typename Arch>
simd::batch<T, Arch> toms748(const Function& f,
                             const simd::batch<T, Arch> lower_bound,
                             const simd::batch<T, Arch> upper_bound,
                             const simd::batch<T, Arch> f_at_lower_bound,
                             const simd::batch<T, Arch> f_at_upper_bound,
                             const T absolute_tolerance,
                             const T relative_tolerance,
                             const size_t max_iterations = 100) {
  ASSERT(relative_tolerance > std::numeric_limits<T>::epsilon(),
         "The relative tolerance is too small.");

  std::size_t max_iters = max_iterations;

  // This solver requires tol to be passed as a termination condition. This
  // termination condition is equivalent to the convergence criteria used by the
  // GSL
  const auto tol = [absolute_tolerance, relative_tolerance](
                       const simd::batch<T, Arch>& lhs,
                       const simd::batch<T, Arch>& rhs) {
    return simd::abs(lhs - rhs) <=
           simd::fma(simd::batch<T, Arch>(relative_tolerance),
                     simd::min(simd::abs(lhs), simd::abs(rhs)),
                     simd::batch<T, Arch>(absolute_tolerance));
  };
  auto result = toms748_detail::toms748_solve(f, lower_bound, upper_bound,
                                              f_at_lower_bound,
                                              f_at_upper_bound, tol, max_iters);
  if (max_iters >= max_iterations) {
    throw convergence_error(
        "toms748 reached max iterations without converging");
  }
  return simd::fma(simd::batch<T, Arch>(static_cast<T>(0.5)),
                   (result.second - result.first), result.first);
}

template <typename Function, typename T, typename Arch>
simd::batch<T, Arch> toms748(const Function& f,
                             const simd::batch<T, Arch> lower_bound,
                             const simd::batch<T, Arch> upper_bound,
                             const T absolute_tolerance,
                             const T relative_tolerance,
                             const size_t max_iterations = 100) {
  return toms748(f, lower_bound, upper_bound, f(lower_bound), f(upper_bound),
                 absolute_tolerance, relative_tolerance, max_iterations);
}
#endif  // SPECTRE_USE_SIMD

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the TOMS_748 method.
 *
 * `f` is a unary invokable that takes a `double` which is the current value at
 * which to evaluate `f`. An example is below.
 *
 * \snippet Test_TOMS748.cpp double_root_find
 *
 * The TOMS_748 algorithm searches for a root in the interval [`lower_bound`,
 * `upper_bound`], and will throw if this interval does not bracket a root,
 * i.e. if `f(lower_bound) * f(upper_bound) > 0`.
 *
 * The arguments `f_at_lower_bound` and `f_at_upper_bound` are optional, and
 * are the function values at `lower_bound` and `upper_bound`. These function
 * values are often known because the user typically checks if a root is
 * bracketed before calling `toms748`; passing the function values here saves
 * two function evaluations.
 *
 * See the [Boost](http://www.boost.org/) documentation for more details.
 *
 * \requires Function `f` is invokable with a `double`
 *
 * \throws `convergence_error` if the requested tolerance is not met after
 *                            `max_iterations` iterations.
 */
template <typename Function>
double toms748(const Function& f, const double lower_bound,
               const double upper_bound, const double f_at_lower_bound,
               const double f_at_upper_bound, const double absolute_tolerance,
               const double relative_tolerance,
               const size_t max_iterations = 100) {
  if (f_at_lower_bound * f_at_upper_bound > 0.0) {
    ERROR("Root not bracketed: "
          "f(" << lower_bound << ") = " << f_at_lower_bound << ", "
          "f(" << upper_bound << ") = " << f_at_upper_bound);
  }
  ASSERT(relative_tolerance > std::numeric_limits<double>::epsilon(),
         "The relative tolerance is too small.");

  boost::uintmax_t max_iters = max_iterations;

  // This solver requires tol to be passed as a termination condition. This
  // termination condition is equivalent to the convergence criteria used by the
  // GSL
  auto tol = [absolute_tolerance, relative_tolerance](double lhs, double rhs) {
    return (fabs(lhs - rhs) <=
            absolute_tolerance +
                relative_tolerance * fmin(fabs(lhs), fabs(rhs)));
  };
  // clang-tidy: internal boost warning, can't fix it.
  auto result = boost::math::tools::toms748_solve(  // NOLINT
      f, lower_bound, upper_bound, f_at_lower_bound, f_at_upper_bound, tol,
      max_iters);
  if (max_iters >= max_iterations) {
    throw convergence_error(
        "toms748 reached max iterations without converging");
  }
  return result.first + 0.5 * (result.second - result.first);
}

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the TOMS_748 method,
 * where function values are not supplied at the lower and upper
 * bounds.
 */
template <typename Function>
double toms748(const Function& f, const double lower_bound,
               const double upper_bound, const double absolute_tolerance,
               const double relative_tolerance,
               const size_t max_iterations = 100) {
  return toms748(f, lower_bound, upper_bound, f(lower_bound), f(upper_bound),
                 absolute_tolerance, relative_tolerance, max_iterations);
}

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the TOMS_748 method on each
 * element in a `DataVector`.
 *
 * `f` is a binary invokable that takes a `double` as its first argument and a
 * `size_t` as its second. The `double` is the current value at which to
 * evaluate `f`, and the `size_t` is the current index into the `DataVector`s.
 * Below is an example of how to root find different functions by indexing into
 * a lambda-captured `DataVector` using the `size_t` passed to `f`.
 *
 * \snippet Test_TOMS748.cpp datavector_root_find
 *
 * For each index `i` into the DataVector, the TOMS_748 algorithm searches for a
 * root in the interval [`lower_bound[i]`, `upper_bound[i]`], and will throw if
 * this interval does not bracket a root,
 * i.e. if `f(lower_bound[i], i) * f(upper_bound[i], i) > 0`.
 *
 * See the [Boost](http://www.boost.org/) documentation for more details.
 *
 * \requires Function `f` be callable with a `double` and a `size_t`
 *
 * \throws `convergence_error` if, for any index, the requested tolerance is not
 * met after `max_iterations` iterations.
 */
template <typename Function>
DataVector toms748(const Function& f, const DataVector& lower_bound,
                   const DataVector& upper_bound,
                   const double absolute_tolerance,
                   const double relative_tolerance,
                   const size_t max_iterations = 100) {
  DataVector result_vector{lower_bound.size()};
  for (size_t i = 0; i < result_vector.size(); ++i) {
    result_vector[i] = toms748(
        [&f, i](double x) { return f(x, i); }, lower_bound[i], upper_bound[i],
        absolute_tolerance, relative_tolerance, max_iterations);
  }
  return result_vector;
}

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Finds the root of the function `f` with the TOMS_748 method on each
 * element in a `DataVector`, where function values are supplied at the lower
 * and upper bounds.
 *
 * Supplying function values is an optimization that saves two
 * function calls per point.  The function values are often available
 * because one often checks if the root is bracketed before calling `toms748`.
 */
template <typename Function>
DataVector toms748(const Function& f, const DataVector& lower_bound,
                   const DataVector& upper_bound,
                   const DataVector& f_at_lower_bound,
                   const DataVector& f_at_upper_bound,
                   const double absolute_tolerance,
                   const double relative_tolerance,
                   const size_t max_iterations = 100) {
  DataVector result_vector{lower_bound.size()};
  for (size_t i = 0; i < result_vector.size(); ++i) {
    result_vector[i] =
        toms748([&f, i](double x) { return f(x, i); }, lower_bound[i],
                upper_bound[i], f_at_lower_bound[i], f_at_upper_bound[i],
                absolute_tolerance, relative_tolerance, max_iterations);
  }
  return result_vector;
}

}  // namespace RootFinder
