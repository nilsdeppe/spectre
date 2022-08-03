// Distributed under the MIT License.
// See LICENSE.txt for details.

/*

#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"

#include <limits>

#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Simd/Simd.hpp"

namespace RootFinder {
#ifdef SPECTRE_USE_SIMD
namespace toms748_detail {
template <typename T>
simd::batch<T> safe_div(const simd::batch<T>& num, const simd::batch<T>& denom,
                        const simd::batch<T>& r) {
  // return num / denom without overflow, return r if overflow would occur.
  const auto mask = fabs(denom) < (static_cast<T>(1));
  if (UNLIKELY(simd::any(mask))) {
    return simd::select(
        mask and fabs(denom * std::numeric_limits<T>::max()) <= fabs(num), r,
        num / denom);
  }
  return num / denom;
}

template <typename T>
simd::batch<T> secant_interpolate(const simd::batch<T>& a,
                                  const simd::batch<T>& b,
                                  const simd::batch<T>& fa,
                                  const simd::batch<T>& fb) {
  //
  // Performs standard secant interpolation of [a,b] given
  // function evaluations f(a) and f(b).  Performs a bisection
  // if secant interpolation would leave us very close to either
  // a or b.  Rationale: we only call this function when at least
  // one other form of interpolation has already failed, so we know
  // that the function is unlikely to be smooth with a root very
  // close to a or b.
  //

  const T tol = std::numeric_limits<T>::epsilon() * static_cast<T>(5);
  const simd::batch<T> c = a - (fa / (fb - fa)) * (b - a);
  return simd::select((c <= a + fabs(a) * tol) or (c >= b - fabs(b) * tol),
                      static_cast<T>(0.5) * (a + b), c);
}

template <typename T>
simd::batch<T> quadratic_interpolate(
    const simd::batch<T>& a, const simd::batch<T>& b, const simd::batch<T>& d,
    const simd::batch<T>& fa, const simd::batch<T>& fb,
    const simd::batch<T>& fd, const unsigned count) {
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
  const simd::batch<T> B =
      safe_div(fb - fa, b - a, simd::batch<T>(std::numeric_limits<T>::max()));
  simd::batch<T> A =
      safe_div(fd - fb, d - b, simd::batch<T>(std::numeric_limits<T>::max()));
  A = safe_div(A - B, d - a, simd::batch<T>(0));

  const auto secant_failure_mask = A == static_cast<T>(0);
  simd::batch<T> result_secant{};
  if (UNLIKELY(simd::any(secant_failure_mask))) {
    // failure to determine coefficients, try a secant step:
    result_secant = secant_interpolate(a, b, fa, fb);
    if (UNLIKELY(simd::all(secant_failure_mask))) {
      return result_secant;
    }
  }

  // Determine the starting point of the Newton steps:
  simd::batch<T> c =
      simd::select(simd::sign(A) * simd::sign(fa) > static_cast<T>(0) and
                       A != static_cast<T>(0) and fa != static_cast<T>(0),
                   a, b);

  // Take the Newton steps:
  for (unsigned i = 1; i <= count; ++i) {
    c -= safe_div(fa + (B + A * (c - b)) * (c - a),
                  B + A * (static_cast<T>(2) * c - a - b),
                  static_cast<T>(1) + c - a);
  }
  if (const auto mask = (c <= a) or (c >= b); simd::any(mask)) {
    // Oops, failure, try a secant step:
    c = simd::select(mask, secant_interpolate(a, b, fa, fb), c);
  }
  return simd::select(secant_failure_mask, result_secant, c);
}

template <typename T>
simd::batch<T> cubic_interpolate(
    const simd::batch<T>& a, const simd::batch<T>& b, const simd::batch<T>& d,
    const simd::batch<T>& e, const simd::batch<T>& fa, const simd::batch<T>& fb,
    const simd::batch<T>& fd, const simd::batch<T>& fe) {
  // Uses inverse cubic interpolation of f(x) at points
  // [a,b,d,e] to obtain an approximate root of f(x).
  // Points d and e lie outside the interval [a,b]
  // and are the third and forth best approximations
  // to the root that we have found so far.
  //
  // Note: this does not guarantee to find a root
  // inside [a, b], so we fall back to quadratic
  // interpolation in case of an erroneous result.
  const simd::batch<T> q11 = (d - e) * fd / (fe - fd);
  const simd::batch<T> q21 = (b - d) * fb / (fd - fb);
  const simd::batch<T> q31 = (a - b) * fa / (fb - fa);
  const simd::batch<T> d21 = (b - d) * fd / (fd - fb);
  const simd::batch<T> d31 = (a - b) * fb / (fb - fa);

  const simd::batch<T> q22 = (d21 - q11) * fb / (fe - fb);
  const simd::batch<T> q32 = (d31 - q21) * fa / (fd - fa);
  const simd::batch<T> d32 = (d31 - q21) * fd / (fd - fa);
  const simd::batch<T> q33 = (d32 - q22) * fa / (fe - fa);
  simd::batch<T> c = q31 + q32 + q33 + a;

  if (const auto mask = (c <= a) or (c >= b); simd::any(mask)) {
    // Out of bounds step, fall back to quadratic interpolation:
    c = simd::select(mask, quadratic_interpolate(a, b, d, fa, fb, fd, 3), c);
  }

  return c;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template simd::batch<DTYPE(data)> secant_interpolate(                        \
      const simd::batch<DTYPE(data)>& a, const simd::batch<DTYPE(data)>& b,    \
      const simd::batch<DTYPE(data)>& fa, const simd::batch<DTYPE(data)>& fb); \
  template simd::batch<DTYPE(data)> quadratic_interpolate(                     \
      const simd::batch<DTYPE(data)>& a, const simd::batch<DTYPE(data)>& b,    \
      const simd::batch<DTYPE(data)>& d, const simd::batch<DTYPE(data)>& fa,   \
      const simd::batch<DTYPE(data)>& fb, const simd::batch<DTYPE(data)>& fd,  \
      const unsigned count);                                                   \
  template simd::batch<DTYPE(data)> cubic_interpolate(                         \
      const simd::batch<DTYPE(data)>& a, const simd::batch<DTYPE(data)>& b,    \
      const simd::batch<DTYPE(data)>& d, const simd::batch<DTYPE(data)>& e,    \
      const simd::batch<DTYPE(data)>& fa, const simd::batch<DTYPE(data)>& fb,  \
      const simd::batch<DTYPE(data)>& fd, const simd::batch<DTYPE(data)>& fe);

GENERATE_INSTANTIATIONS(INSTANTIATION, (float, double))

#undef INSTANTIATION
#undef DTYPE
}  // namespace toms748_detail
#endif  // SPECTRE_USE_SIMD
}  // namespace RootFinder
*/
