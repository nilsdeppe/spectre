// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace {
// ---------------------------------------------------------------------------
// Helpers for ill-conditioned accuracy tests
// ---------------------------------------------------------------------------

// Compute the max relative error between two square tensors (double, any dim).
// Uses the max absolute value of `reference` as the denominator.
template <typename TensorType>
double max_rel_error(const TensorType& computed, const TensorType& reference) {
  double ref_scale = 0.0;
  for (size_t i = 0; i < TensorType::size(); ++i) {
    ref_scale = std::max(ref_scale, std::abs(reference[i]));
  }
  if (ref_scale == 0.0) {
    return 0.0;
  }
  double err = 0.0;
  for (size_t i = 0; i < TensorType::size(); ++i) {
    err = std::max(err, std::abs(computed[i] - reference[i]));
  }
  return err / ref_scale;
}

// Invert a 4x4 generic matrix using long double Cramer's rule to get a
// high-precision reference.  Returns the inverse; det is also set.
tnsr::IJ<double, 4, Frame::Grid> reference_inverse_4d_generic(
    const tnsr::ij<double, 4, Frame::Grid>& t, double& det_out) {
  // Promote to long double for extra precision.
  const long double t00 = get<0, 0>(t), t01 = get<0, 1>(t), t02 = get<0, 2>(t),
                    t03 = get<0, 3>(t);
  const long double t10 = get<1, 0>(t), t11 = get<1, 1>(t), t12 = get<1, 2>(t),
                    t13 = get<1, 3>(t);
  const long double t20 = get<2, 0>(t), t21 = get<2, 1>(t), t22 = get<2, 2>(t),
                    t23 = get<2, 3>(t);
  const long double t30 = get<3, 0>(t), t31 = get<3, 1>(t), t32 = get<3, 2>(t),
                    t33 = get<3, 3>(t);

  const long double m23_23 = t22 * t33 - t23 * t32;
  const long double m23_13 = t21 * t33 - t23 * t31;
  const long double m23_03 = t20 * t33 - t23 * t30;
  const long double m23_12 = t21 * t32 - t22 * t31;
  const long double m23_02 = t20 * t32 - t22 * t30;
  const long double m23_01 = t20 * t31 - t21 * t30;
  const long double m13_23 = t12 * t33 - t13 * t32;
  const long double m13_13 = t11 * t33 - t13 * t31;
  const long double m13_03 = t10 * t33 - t13 * t30;
  const long double m13_12 = t11 * t32 - t12 * t31;
  const long double m13_02 = t10 * t32 - t12 * t30;
  const long double m13_01 = t10 * t31 - t11 * t30;
  const long double m12_23 = t12 * t23 - t13 * t22;
  const long double m12_13 = t11 * t23 - t13 * t21;
  const long double m12_03 = t10 * t23 - t13 * t20;
  const long double m12_12 = t11 * t22 - t12 * t21;
  const long double m12_02 = t10 * t22 - t12 * t20;
  const long double m12_01 = t10 * t21 - t11 * t20;

  const long double det = t00 * (t11 * m23_23 - t12 * m23_13 + t13 * m23_12) -
                          t01 * (t10 * m23_23 - t12 * m23_03 + t13 * m23_02) +
                          t02 * (t10 * m23_13 - t11 * m23_03 + t13 * m23_01) -
                          t03 * (t10 * m23_12 - t11 * m23_02 + t12 * m23_01);
  det_out = static_cast<double>(det);
  const long double ood = 1.0L / det;

  tnsr::IJ<double, 4, Frame::Grid> inv{};
  get<0, 0>(inv) =
      static_cast<double>((t11 * m23_23 - t12 * m23_13 + t13 * m23_12) * ood);
  get<0, 1>(inv) =
      static_cast<double>(-(t01 * m23_23 - t02 * m23_13 + t03 * m23_12) * ood);
  get<0, 2>(inv) =
      static_cast<double>((t01 * m13_23 - t02 * m13_13 + t03 * m13_12) * ood);
  get<0, 3>(inv) =
      static_cast<double>(-(t01 * m12_23 - t02 * m12_13 + t03 * m12_12) * ood);
  get<1, 0>(inv) =
      static_cast<double>(-(t10 * m23_23 - t12 * m23_03 + t13 * m23_02) * ood);
  get<1, 1>(inv) =
      static_cast<double>((t00 * m23_23 - t02 * m23_03 + t03 * m23_02) * ood);
  get<1, 2>(inv) =
      static_cast<double>(-(t00 * m13_23 - t02 * m13_03 + t03 * m13_02) * ood);
  get<1, 3>(inv) =
      static_cast<double>((t00 * m12_23 - t02 * m12_03 + t03 * m12_02) * ood);
  get<2, 0>(inv) =
      static_cast<double>((t10 * m23_13 - t11 * m23_03 + t13 * m23_01) * ood);
  get<2, 1>(inv) =
      static_cast<double>(-(t00 * m23_13 - t01 * m23_03 + t03 * m23_01) * ood);
  get<2, 2>(inv) =
      static_cast<double>((t00 * m13_13 - t01 * m13_03 + t03 * m13_01) * ood);
  get<2, 3>(inv) =
      static_cast<double>(-(t00 * m12_13 - t01 * m12_03 + t03 * m12_01) * ood);
  get<3, 0>(inv) =
      static_cast<double>(-(t10 * m23_12 - t11 * m23_02 + t12 * m23_01) * ood);
  get<3, 1>(inv) =
      static_cast<double>((t00 * m23_12 - t01 * m23_02 + t02 * m23_01) * ood);
  get<3, 2>(inv) =
      static_cast<double>(-(t00 * m13_12 - t01 * m13_02 + t02 * m13_01) * ood);
  get<3, 3>(inv) =
      static_cast<double>((t00 * m12_12 - t01 * m12_02 + t02 * m12_01) * ood);
  return inv;
}

// Test A: 4x4 generic matrix where the top-left 2x2 block P is nearly singular
// but the full matrix has det(A) ~ 2.
//
//   A = | 1+d  1   1   0 |
//       | 1    1   0   1 |   with d = 1e-12
//       | 1    0   2   0 |
//       | 0    1   0   2 |
//
// det(P) = (1+d)*1 - 1*1 = d = 1e-12 << 1.
// det(S) = 2*2 - 0*0 = 4  (well-conditioned bottom-right block).
//
// BlockTopLeft divides by det(P) first, amplifying roundoff in all
// intermediate quantities.  BlockBottomRight and Analytic avoid this.
void test_4x4_near_singular_top_left_block() {
  const double d = 1.0e-12;
  tnsr::ij<double, 4, Frame::Grid> t{};
  get<0, 0>(t) = 1.0 + d;
  get<0, 1>(t) = 1.0;
  get<0, 2>(t) = 1.0;
  get<0, 3>(t) = 0.0;
  get<1, 0>(t) = 1.0;
  get<1, 1>(t) = 1.0;
  get<1, 2>(t) = 0.0;
  get<1, 3>(t) = 1.0;
  get<2, 0>(t) = 1.0;
  get<2, 1>(t) = 0.0;
  get<2, 2>(t) = 2.0;
  get<2, 3>(t) = 0.0;
  get<3, 0>(t) = 0.0;
  get<3, 1>(t) = 1.0;
  get<3, 2>(t) = 0.0;
  get<3, 3>(t) = 2.0;

  double ref_det = 0.0;
  const auto ref_inv = reference_inverse_4d_generic(t, ref_det);

  const auto [det_tl, inv_tl] =
      determinant_and_inverse<InversionMethod::BlockTopLeft>(t);
  const auto [det_br, inv_br] =
      determinant_and_inverse<InversionMethod::BlockBottomRight>(t);
  const auto [det_an, inv_an] =
      determinant_and_inverse<InversionMethod::Analytic>(t);

  const double err_tl = max_rel_error(inv_tl, ref_inv);
  const double err_br = max_rel_error(inv_br, ref_inv);
  const double err_an = max_rel_error(inv_an, ref_inv);

  // BlockTopLeft should be dramatically worse than the alternatives because
  // it divides by det(P) = 1e-12 in the first step.
  // BlockBottomRight and Analytic should have comparable accuracy to each
  // other (both avoid the near-singular top-left block).
  CHECK(err_tl > err_an * 1.0e4);
  CHECK(err_tl > err_br * 1.0e4);
  // Analytic and BlockBottomRight should be within a small constant of each
  // other (both use well-conditioned intermediate steps).
  CHECK(err_br < err_tl * 1.0e-3);
  CHECK(err_an < err_tl * 1.0e-3);
  // The determinants from the better methods should be close to the reference.
  CHECK(std::abs(det_br.get() - ref_det) <
        1.0e6 * std::numeric_limits<double>::epsilon() * std::abs(ref_det));
  CHECK(std::abs(det_an.get() - ref_det) <
        1.0e6 * std::numeric_limits<double>::epsilon() * std::abs(ref_det));
}

// Test B: 3x3 Hilbert matrix (well-known ill-conditioned matrix).
//
//   H_3 = | 1    1/2  1/3 |
//         | 1/2  1/3  1/4 |
//         | 1/3  1/4  1/5 |
//
// kappa(H_3) ~ 524.  The exact inverse has large integer entries.
// Cramer's rule error is O(3 * 524 * u) ~ 1.7e-13.
// This test verifies that the Analytic method achieves the expected accuracy.
void test_3x3_hilbert() {
  tnsr::ii<double, 3, Frame::Grid> t{};
  get<0, 0>(t) = 1.0;
  get<0, 1>(t) = 1.0 / 2.0;
  get<0, 2>(t) = 1.0 / 3.0;
  get<1, 1>(t) = 1.0 / 3.0;
  get<1, 2>(t) = 1.0 / 4.0;
  get<2, 2>(t) = 1.0 / 5.0;

  // The exact inverse of H_3 has known integer entries:
  //   | 9   -36    30 |
  //   | -36  192  -180|
  //   | 30  -180  180 |
  tnsr::II<double, 3, Frame::Grid> exact_inv{};
  get<0, 0>(exact_inv) = 9.0;
  get<0, 1>(exact_inv) = -36.0;
  get<0, 2>(exact_inv) = 30.0;
  get<1, 1>(exact_inv) = 192.0;
  get<1, 2>(exact_inv) = -180.0;
  get<2, 2>(exact_inv) = 180.0;

  const auto [det, inv] = determinant_and_inverse<InversionMethod::Analytic>(t);

  // kappa ~ 524; expected relative error ~ 3 * 524 * eps ~ 1.7e-13.
  // Use a generous threshold of 1e-11 to allow for constant factors.
  CHECK(max_rel_error(inv, exact_inv) < 1.0e-11);
  CHECK(approx(det.get()) == 1.0 / 2160.0);

  // Refined: one Newton-Schulz step quadratically improves accuracy.
  // For kappa ~ 524, err_refined ~ err_analytic^2 * kappa ~ 1e-23, which is
  // far below machine epsilon, so the result should be at machine precision.
  const auto [det_r, inv_r] =
      determinant_and_inverse<InversionMethod::Refined>(t);
  CHECK(max_rel_error(inv_r, exact_inv) < 1.0e-13);
  CHECK(approx(det_r.get()) == 1.0 / 2160.0);
}

// Test C: 3x3 symmetric matrix with large eigenvalue spread.
// Constructed as Q^T * diag(1, 1, eps) * Q where Q is a rotation and
// eps = 1e-10, giving kappa(A) = 1e10.
//
// Cramer's rule gives ~6 correct digits at best.
// This test verifies that Analytic achieves the accuracy bound O(3*kappa*u)
// by comparing against a long-double reference.
void test_3x3_large_eigenvalue_spread() {
  // Build A = Q^T * D * Q where Q is a 45-degree rotation in the (0,2) plane
  // and D = diag(1, 1, 1e-10).
  const double eps = 1.0e-10;
  const double c = std::sqrt(0.5);  // cos(45 deg)
  const double s = std::sqrt(0.5);  // sin(45 deg)

  // Q = rotation in (0,2)-plane: Q_{00}=c, Q_{02}=-s, Q_{20}=s, Q_{22}=c
  // A_{ij} = sum_k Q_{ki} * D_{kk} * Q_{kj}
  // With D = diag(1, 1, eps):
  //   A_{00} = Q_{00}^2 * 1 + Q_{10}^2 * 1 + Q_{20}^2 * eps
  //          = c^2 + 0 + s^2 * eps = 0.5 + 0.5 * eps
  //   A_{01} = Q_{00}*Q_{01}*1 + ... (Q_{01}=0, Q_{21}=0) = 0
  //   A_{02} = Q_{00}*Q_{02}*1 + Q_{20}*Q_{22}*eps
  //          = c*(-s) + s*c*eps = -c*s*(1 - eps)
  //   A_{11} = Q_{11}^2 = 1  (Q_{01}=Q_{21}=0, Q_{11}=1)
  //   A_{12} = Q_{11}*Q_{12}*1 + ... (Q_{12}=0) = 0
  //   A_{22} = Q_{02}^2*1 + Q_{22}^2*eps = s^2 + c^2*eps = 0.5 + 0.5*eps
  tnsr::ii<double, 3, Frame::Grid> t{};
  get<0, 0>(t) = c * c + s * s * eps;
  get<0, 1>(t) = 0.0;
  get<0, 2>(t) = -c * s * (1.0 - eps);
  get<1, 1>(t) = 1.0;
  get<1, 2>(t) = 0.0;
  get<2, 2>(t) = s * s + c * c * eps;

  // Long-double reference inverse via Cramer's rule.
  const long double lc = c, ls = s, leps = eps;
  const long double lt00 = lc * lc + ls * ls * leps;
  const long double lt01 = 0.0L;
  const long double lt02 = -lc * ls * (1.0L - leps);
  const long double lt11 = 1.0L;
  const long double lt12 = 0.0L;
  const long double lt22 = ls * ls + lc * lc * leps;
  const long double la = lt11 * lt22 - lt12 * lt12;
  const long double lb = lt12 * lt02 - lt01 * lt22;
  const long double lc2 = lt01 * lt12 - lt11 * lt02;
  const long double ldet = lt00 * la + lt01 * lb + lt02 * lc2;
  const long double lood = 1.0L / ldet;
  tnsr::II<double, 3, Frame::Grid> ref_inv{};
  get<0, 0>(ref_inv) = static_cast<double>((lt11 * lt22 - lt12 * lt12) * lood);
  get<0, 1>(ref_inv) = static_cast<double>((lt12 * lt02 - lt22 * lt01) * lood);
  get<0, 2>(ref_inv) = static_cast<double>((lt01 * lt12 - lt02 * lt11) * lood);
  get<1, 1>(ref_inv) = static_cast<double>((lt22 * lt00 - lt02 * lt02) * lood);
  get<1, 2>(ref_inv) = static_cast<double>((lt02 * lt01 - lt00 * lt12) * lood);
  get<2, 2>(ref_inv) = static_cast<double>((lt00 * lt11 - lt01 * lt01) * lood);

  const auto [det, inv] = determinant_and_inverse<InversionMethod::Analytic>(t);

  // kappa = 1e10. Expected relative error ~ 3 * 1e10 * eps_machine ~ 3e-6.
  // Threshold is 1e-4 to account for constant factors and allow the test to
  // be robust, while still catching catastrophic cancellation.
  CHECK(max_rel_error(inv, ref_inv) < 1.0e-4);
}

// Test that the Refined method improves accuracy for the generic
// (non-symmetric) 3x3 path.  Uses the 3x3 Hilbert matrix viewed as a generic
// (Symmetry<2,1>) tensor, where the exact inverse entries are known integers.
void test_refined_generic_3x3() {
  // Build H_3 as a generic (non-symmetric) tensor so Symmetry<2,1> path fires.
  using GenericTensor =
      Tensor<double, tmpl::integral_list<int32_t, 2, 1>,
             index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                        SpatialIndex<3, UpLo::Lo, Frame::Grid>>>;
  GenericTensor t{};
  get<0, 0>(t) = 1.0;
  get<0, 1>(t) = 1.0 / 2.0;
  get<0, 2>(t) = 1.0 / 3.0;
  get<1, 0>(t) = 1.0 / 2.0;
  get<1, 1>(t) = 1.0 / 3.0;
  get<1, 2>(t) = 1.0 / 4.0;
  get<2, 0>(t) = 1.0 / 3.0;
  get<2, 1>(t) = 1.0 / 4.0;
  get<2, 2>(t) = 1.0 / 5.0;

  // Exact inverse of H_3 has known integer entries (same as symmetric case
  // but stored as a fully general 3x3 matrix):
  using GenericInv = Tensor<double, tmpl::integral_list<int32_t, 2, 1>,
                            index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                                       SpatialIndex<3, UpLo::Up, Frame::Grid>>>;
  GenericInv exact_inv{};
  get<0, 0>(exact_inv) = 9.0;
  get<0, 1>(exact_inv) = -36.0;
  get<0, 2>(exact_inv) = 30.0;
  get<1, 0>(exact_inv) = -36.0;
  get<1, 1>(exact_inv) = 192.0;
  get<1, 2>(exact_inv) = -180.0;
  get<2, 0>(exact_inv) = 30.0;
  get<2, 1>(exact_inv) = -180.0;
  get<2, 2>(exact_inv) = 180.0;

  const auto [det_an, inv_an] =
      determinant_and_inverse<InversionMethod::Analytic>(t);
  const auto [det_r, inv_r] =
      determinant_and_inverse<InversionMethod::Refined>(t);

  // Analytic: kappa ~ 524, rel error < 1e-11.
  CHECK(max_rel_error(inv_an, exact_inv) < 1.0e-11);
  // Refined quadratically improves: err ~ err_analytic^2 * kappa << eps.
  CHECK(max_rel_error(inv_r, exact_inv) < 1.0e-13);
  CHECK(approx(det_r.get()) == 1.0 / 2160.0);
}

// Test that the Refined path (with equilibration) handles 3x3 SPD matrices
// whose diagonal entries span many orders of magnitude.
//
// Matrix: A = D * H * D, D = diag(1e3, 1, 1e-3), H = 3x3 Hilbert matrix.
// The diagonal entries of A span 13 orders of magnitude (A_00=1e6, A_22=2e-7).
// The exact inverse is inv(A) = D^{-1} * inv(H) * D^{-1}.
void test_3x3_spd_equilibration() {
  tnsr::ii<double, 3, Frame::Grid> t{};
  get<0, 0>(t) = 1.0e6;      // D_0^2 * H_00 = (1e3)^2 * 1
  get<0, 1>(t) = 5.0e2;      // D_0 * H_01 * D_1 = 1e3 * 0.5 * 1
  get<0, 2>(t) = 1.0 / 3.0;  // D_0 * H_02 * D_2 = 1e3 * (1/3) * 1e-3
  get<1, 1>(t) = 1.0 / 3.0;  // D_1^2 * H_11 = 1 * (1/3)
  get<1, 2>(t) = 2.5e-4;     // D_1 * H_12 * D_2 = 1 * 0.25 * 1e-3
  get<2, 2>(t) = 2.0e-7;     // D_2^2 * H_22 = (1e-3)^2 * 0.2

  // inv(A) = D^{-1} * inv(H) * D^{-1}, D^{-1} = diag(1e-3, 1, 1e3)
  // inv(H) = [[9, -36, 30], [-36, 192, -180], [30, -180, 180]]
  tnsr::II<double, 3, Frame::Grid> exact_inv{};
  get<0, 0>(exact_inv) = 9.0e-6;   // 1e-3 * 9 * 1e-3
  get<0, 1>(exact_inv) = -3.6e-2;  // 1e-3 * (-36) * 1
  get<0, 2>(exact_inv) = 3.0e1;    // 1e-3 * 30 * 1e3
  get<1, 1>(exact_inv) = 1.92e2;   // 1 * 192 * 1
  get<1, 2>(exact_inv) = -1.8e5;   // 1 * (-180) * 1e3
  get<2, 2>(exact_inv) = 1.8e8;    // 1e3 * 180 * 1e3

  // Analytic Cramer on the unscaled matrix; condition number of A satisfies
  // kappa_inf(A) = ||A||_inf * ||inv(A)||_inf ~ 1e6 * 1.8e8 = 1.8e14, so
  // the relative error could in principle reach O(kappa_inf * u) ~ 1e-2.
  // In practice cancellation is moderate (O(20x)), so the actual error is
  // closer to O(20 * u) ~ 2e-15; we assert a conservative bound of 1e-11.
  const auto [det_an, inv_an] =
      determinant_and_inverse<InversionMethod::Analytic>(t);
  CHECK(max_rel_error(inv_an, exact_inv) < 1.0e-11);

  // Refined equilibrates (D_eq = diag(1/sqrt(A_ii))), reducing the effective
  // kappa to that of the scaled matrix (~524), then applies Newton-Schulz.
  // The result should be near machine precision.
  const auto [det_r, inv_r] =
      determinant_and_inverse<InversionMethod::Refined>(t);
  // Refined must be at least as accurate as Analytic.
  CHECK(max_rel_error(inv_r, exact_inv) <= max_rel_error(inv_an, exact_inv));
  // Refined should achieve near-machine-precision.
  CHECK(max_rel_error(inv_r, exact_inv) < 1.0e-13);
  // det(A) = det(D)^2 * det(H) = 1 * 1/2160
  CHECK(approx(det_r.get()) == 1.0 / 2160.0);
}

// Test Refined on a DataVector input to verify the vectorized path works.
// Uses a 2-point DataVector where each point holds the 3x3 Hilbert matrix.
// The exact inverse is known (integer entries), and both points should give
// near-machine-precision results from equilibration + Newton-Schulz.
void test_refined_datavector_3x3() {
  tnsr::ii<DataVector, 3, Frame::Grid> t{};
  // Both grid points hold the same Hilbert matrix H_3.
  get<0, 0>(t) = DataVector({1.0, 1.0});
  get<0, 1>(t) = DataVector({0.5, 0.5});
  get<0, 2>(t) = DataVector({1.0 / 3.0, 1.0 / 3.0});
  get<1, 1>(t) = DataVector({1.0 / 3.0, 1.0 / 3.0});
  get<1, 2>(t) = DataVector({0.25, 0.25});
  get<2, 2>(t) = DataVector({0.2, 0.2});

  const auto [det_r, inv_r] =
      determinant_and_inverse<InversionMethod::Refined>(t);

  // Exact inverse of H_3: [[9,-36,30],[-36,192,-180],[30,-180,180]]
  // Both points should yield near-machine-precision results.
  const DataVector expect_00({9.0, 9.0});
  const DataVector expect_01({-36.0, -36.0});
  const DataVector expect_02({30.0, 30.0});
  const DataVector expect_11({192.0, 192.0});
  const DataVector expect_12({-180.0, -180.0});
  const DataVector expect_22({180.0, 180.0});
  const DataVector expect_det({1.0 / 2160.0, 1.0 / 2160.0});

  const DataVector& comp_00 = get<0, 0>(inv_r);
  const DataVector& comp_01 = get<0, 1>(inv_r);
  const DataVector& comp_02 = get<0, 2>(inv_r);
  const DataVector& comp_11 = get<1, 1>(inv_r);
  const DataVector& comp_12 = get<1, 2>(inv_r);
  const DataVector& comp_22 = get<2, 2>(inv_r);
  const DataVector& comp_det = get(det_r);
  CHECK_ITERABLE_APPROX(comp_00, expect_00);
  CHECK_ITERABLE_APPROX(comp_01, expect_01);
  CHECK_ITERABLE_APPROX(comp_02, expect_02);
  CHECK_ITERABLE_APPROX(comp_11, expect_11);
  CHECK_ITERABLE_APPROX(comp_12, expect_12);
  CHECK_ITERABLE_APPROX(comp_22, expect_22);
  CHECK_ITERABLE_APPROX(comp_det, expect_det);
}

// In the spirit of the Tensor type aliases, but for a rank-2 Tensor with each
// index in a different frame. If Fr1 == Fr2, then this reduces to tnsr::iJ.
template <typename DataType, size_t Dim, typename Fr1, typename Fr2>
using tnsr_iJ = Tensor<DataType, tmpl::integral_list<int32_t, 2, 1>,
                       index_list<SpatialIndex<Dim, UpLo::Lo, Fr1>,
                                  SpatialIndex<Dim, UpLo::Up, Fr2>>>;

template <typename TensorType>
void verify_det_and_inv_1d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == 2.0);
  CHECK((get<0, 0>(det_inv.second)) == 0.5);
}

template <typename TensorType>
void verify_det_and_inv_generic_2d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<1, 0>(t) = 4.0;
  get<1, 1>(t) = 5.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == -2.0);
  CHECK((get<0, 0>(det_inv.second)) == -2.5);
  CHECK((get<0, 1>(det_inv.second)) == 1.5);
  CHECK((get<1, 0>(det_inv.second)) == 2.0);
  CHECK((get<1, 1>(det_inv.second)) == -1.0);
}

template <typename TensorType>
void verify_det_and_inv_generic_3d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<0, 2>(t) = 6.0;
  get<1, 0>(t) = 4.0;
  get<1, 1>(t) = 5.0;
  get<1, 2>(t) = 7.0;
  get<2, 0>(t) = 8.0;
  get<2, 1>(t) = 9.0;
  get<2, 2>(t) = 10.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == -2.0);
  CHECK((get<0, 0>(det_inv.second)) == 6.5);
  CHECK((get<0, 1>(det_inv.second)) == -12.0);
  CHECK((get<0, 2>(det_inv.second)) == 4.5);
  CHECK((get<1, 0>(det_inv.second)) == -8.0);
  CHECK((get<1, 1>(det_inv.second)) == 14.0);
  CHECK((get<1, 2>(det_inv.second)) == -5.0);
  CHECK((get<2, 0>(det_inv.second)) == 2.0);
  CHECK((get<2, 1>(det_inv.second)) == -3.0);
  CHECK((get<2, 2>(det_inv.second)) == 1.0);
}

template <typename TensorType>
void verify_det_and_inv_generic_4d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<0, 2>(t) = 6.0;
  get<0, 3>(t) = 11.0;
  get<1, 0>(t) = 4.0;
  get<1, 1>(t) = 5.0;
  get<1, 2>(t) = 7.0;
  get<1, 3>(t) = 12.0;
  get<2, 0>(t) = 8.0;
  get<2, 1>(t) = 9.0;
  get<2, 2>(t) = 10.0;
  get<2, 3>(t) = 13.0;
  get<3, 0>(t) = 14.0;
  get<3, 1>(t) = 15.0;
  get<3, 2>(t) = 16.0;
  get<3, 3>(t) = 17.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == -8.0);
  CHECK((get<0, 0>(det_inv.second)) == -4.0);
  CHECK((get<0, 1>(det_inv.second)) == 9.0);
  CHECK((get<0, 2>(det_inv.second)) == -9.5);
  CHECK((get<0, 3>(det_inv.second)) == 3.5);
  CHECK((get<1, 0>(det_inv.second)) == 3.25);
  CHECK((get<1, 1>(det_inv.second)) == -8.5);
  CHECK((get<1, 2>(det_inv.second)) == 10.0);
  CHECK((get<1, 3>(det_inv.second)) == -3.75);
  CHECK((get<2, 0>(det_inv.second)) == 1.25);
  CHECK((get<2, 1>(det_inv.second)) == -1.5);
  CHECK((get<2, 2>(det_inv.second)) == 0.0);
  CHECK((get<2, 3>(det_inv.second)) == 0.25);
  CHECK((get<3, 0>(det_inv.second)) == -0.75);
  CHECK((get<3, 1>(det_inv.second)) == 1.5);
  CHECK((get<3, 2>(det_inv.second)) == -1.0);
  CHECK((get<3, 3>(det_inv.second)) == 0.25);
}

template <typename TensorType>
void verify_det_and_inv_symmetric_2d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<1, 1>(t) = 5.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == 1.0);
  CHECK((get<0, 0>(det_inv.second)) == 5.0);
  CHECK((get<0, 1>(det_inv.second)) == -3.0);
  CHECK((get<1, 0>(det_inv.second)) == -3.0);
  CHECK((get<1, 1>(det_inv.second)) == 2.0);
}

template <typename TensorType>
void verify_det_and_inv_symmetric_3d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<0, 2>(t) = 6.0;
  get<1, 1>(t) = 5.0;
  get<1, 2>(t) = 7.0;
  get<2, 2>(t) = 10.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == -16.0);
  CHECK((get<0, 0>(det_inv.second)) == -0.0625);
  CHECK((get<0, 1>(det_inv.second)) == -0.75);
  CHECK((get<0, 2>(det_inv.second)) == 0.5625);
  CHECK((get<1, 0>(det_inv.second)) == -0.75);
  CHECK((get<1, 1>(det_inv.second)) == 1.0);
  CHECK((get<1, 2>(det_inv.second)) == -0.25);
  CHECK((get<2, 0>(det_inv.second)) == 0.5625);
  CHECK((get<2, 1>(det_inv.second)) == -0.25);
  CHECK((get<2, 2>(det_inv.second)) == -0.0625);
}

template <typename TensorType>
void verify_det_and_inv_symmetric_4d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<0, 2>(t) = 6.0;
  get<0, 3>(t) = 11.0;
  get<1, 1>(t) = 5.0;
  get<1, 2>(t) = 7.0;
  get<1, 3>(t) = 12.0;
  get<2, 2>(t) = 10.0;
  get<2, 3>(t) = 13.0;
  get<3, 3>(t) = 17.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == -100.0);
  CHECK((get<0, 0>(det_inv.second)) == approx(0.84));
  CHECK((get<0, 1>(det_inv.second)) == approx(-0.94));
  CHECK((get<0, 2>(det_inv.second)) == approx(-0.34));
  CHECK((get<0, 3>(det_inv.second)) == approx(0.38));
  CHECK((get<1, 0>(det_inv.second)) == approx(-0.94));
  CHECK((get<1, 1>(det_inv.second)) == approx(1.04));
  CHECK((get<1, 2>(det_inv.second)) == approx(-0.06));
  CHECK((get<1, 3>(det_inv.second)) == approx(-0.08));
  CHECK((get<2, 0>(det_inv.second)) == approx(-0.34));
  CHECK((get<2, 1>(det_inv.second)) == approx(-0.06));
  CHECK((get<2, 2>(det_inv.second)) == approx(0.84));
  CHECK((get<2, 3>(det_inv.second)) == approx(-0.38));
  CHECK((get<3, 0>(det_inv.second)) == approx(0.38));
  CHECK((get<3, 1>(det_inv.second)) == approx(-0.08));
  CHECK((get<3, 2>(det_inv.second)) == approx(-0.38));
  CHECK((get<3, 3>(det_inv.second)) == approx(0.16));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.DeterminantAndInverse",
                  "[DataStructures][Unit]") {
  // Check that the inverse tensor has the expected index structure -- for an
  // input Tensor of type T^a_b, the inverse should have type T^b_a.
  {
    static_assert(
        std::is_same_v<
            Tensor<double, tmpl::integral_list<int32_t, 2, 1>,
                   index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                              SpatialIndex<2, UpLo::Lo, Frame::Grid>>>,
            decltype(determinant_and_inverse(
                std::declval<
                    Tensor<double, tmpl::integral_list<int32_t, 2, 1>,
                           index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                                      SpatialIndex<2, UpLo::Lo,
                                                   Frame::Inertial>>>>()))::
                second_type>,
        "Inverse tensor has incorrect index structure.");
  }

  // Check paired determinant and inverse for 1x1 through 4x4 tensors, both
  // generic and symmetric, with both spatial and spacetime indices.
  {
    verify_det_and_inv_1d<tnsr::ii<double, 1, Frame::Grid>>();
    verify_det_and_inv_1d<tnsr::ij<double, 1, Frame::Grid>>();
    verify_det_and_inv_1d<tnsr_iJ<double, 1, Frame::Grid, Frame::Inertial>>();

    verify_det_and_inv_symmetric_2d<tnsr::ii<double, 2, Frame::Grid>>();
    verify_det_and_inv_generic_2d<tnsr::ij<double, 2, Frame::Grid>>();
    verify_det_and_inv_generic_2d<
        tnsr_iJ<double, 2, Frame::Grid, Frame::Inertial>>();

    verify_det_and_inv_symmetric_3d<tnsr::ii<double, 3, Frame::Grid>>();
    verify_det_and_inv_generic_3d<tnsr::ij<double, 3, Frame::Grid>>();
    verify_det_and_inv_generic_3d<
        tnsr_iJ<double, 3, Frame::Grid, Frame::Inertial>>();

    verify_det_and_inv_symmetric_4d<tnsr::ii<double, 4, Frame::Grid>>();
    verify_det_and_inv_generic_4d<tnsr::ij<double, 4, Frame::Grid>>();
    verify_det_and_inv_generic_4d<
        tnsr_iJ<double, 4, Frame::Grid, Frame::Inertial>>();

    verify_det_and_inv_symmetric_2d<tnsr::aa<double, 1, Frame::Grid>>();
    verify_det_and_inv_generic_2d<tnsr::ab<double, 1, Frame::Grid>>();

    verify_det_and_inv_symmetric_3d<tnsr::aa<double, 2, Frame::Grid>>();
    verify_det_and_inv_generic_3d<tnsr::ab<double, 2, Frame::Grid>>();

    verify_det_and_inv_symmetric_4d<tnsr::aa<double, 3, Frame::Grid>>();
    verify_det_and_inv_generic_4d<tnsr::ab<double, 3, Frame::Grid>>();
  }

  // Check paired determinant and inverse for a Tensor<DataVector>.
  {
    tnsr::ij<DataVector, 2, Frame::Grid> t{};
    get<0, 0>(t) = DataVector({2.0, 3.0, 5.0, 1.0});
    get<0, 1>(t) = DataVector({3.0, 5.0, 4.0, -1.0});
    get<1, 0>(t) = DataVector({4.0, 2.0, 2.0, 1.0});
    get<1, 1>(t) = DataVector({5.0, 3.0, 2.0, 1.0});
    const auto det_inv = determinant_and_inverse(t);
    CHECK(det_inv.first.get() == DataVector({-2.0, -1.0, 2.0, 2.0}));
    CHECK((get<0, 0>(det_inv.second)) == DataVector({-2.5, -3.0, 1.0, 0.5}));
    CHECK((get<0, 1>(det_inv.second)) == DataVector({1.5, 5.0, -2.0, 0.5}));
    CHECK((get<1, 0>(det_inv.second)) == DataVector({2.0, 2.0, -1.0, -0.5}));
    CHECK((get<1, 1>(det_inv.second)) == DataVector({-1.0, -3.0, 2.5, 0.5}));
  }

  // Check Variables<determinant, inverse> for a Tensor<DataVector>
  {
    {
      // 2D
      using my_tnsr_type = tnsr::ij<DataVector, 2, Frame::Grid>;
      using my_tnsr_inv_type = tnsr::IJ<DataVector, 2, Frame::Grid>;
      using my_tnsr_det_type = Scalar<DataVector>;

      struct MyDetTag : db::SimpleTag {
        static std::string name() { return "DummyDetTag"; }
        using type = my_tnsr_det_type;
      };
      struct MyInvTag : db::SimpleTag {
        static std::string name() { return "DummyInvTag"; }
        using type = my_tnsr_inv_type;
      };

      my_tnsr_type t{};
      get<0, 0>(t) = DataVector({2.0, 3.0, 5.0, 1.0});
      get<0, 1>(t) = DataVector({3.0, 5.0, 4.0, -1.0});
      get<1, 0>(t) = DataVector({4.0, 2.0, 2.0, 1.0});
      get<1, 1>(t) = DataVector({5.0, 3.0, 2.0, 1.0});
      const auto det_inv = determinant_and_inverse<MyDetTag, MyInvTag>(t);
      CHECK(get(get<MyDetTag>(det_inv)) == DataVector({-2.0, -1.0, 2.0, 2.0}));
      CHECK((get<0, 0>(get<MyInvTag>(det_inv))) ==
            DataVector({-2.5, -3.0, 1.0, 0.5}));
      CHECK((get<0, 1>(get<MyInvTag>(det_inv))) ==
            DataVector({1.5, 5.0, -2.0, 0.5}));
      CHECK((get<1, 0>(get<MyInvTag>(det_inv))) ==
            DataVector({2.0, 2.0, -1.0, -0.5}));
      CHECK((get<1, 1>(get<MyInvTag>(det_inv))) ==
            DataVector({-1.0, -3.0, 2.5, 0.5}));
    }
    {
      // 4D
      using my_tnsr_type = tnsr::ij<DataVector, 4, Frame::Grid>;
      using my_tnsr_inv_type = tnsr::IJ<DataVector, 4, Frame::Grid>;
      using my_tnsr_det_type = Scalar<DataVector>;

      struct MyDetTag : db::SimpleTag {
        static std::string name() { return "DummyDetTag"; }
        using type = my_tnsr_det_type;
      };
      struct MyInvTag : db::SimpleTag {
        static std::string name() { return "DummyInvTag"; }
        using type = my_tnsr_inv_type;
      };

      my_tnsr_type t{};
      get<0, 0>(t) = DataVector({2.0, 3.0});
      get<0, 1>(t) = DataVector({3.0, 5.0});
      get<0, 2>(t) = DataVector({4.0, 2.0});
      get<0, 3>(t) = DataVector({5.0, 3.0});
      get<1, 0>(t) = DataVector({2.0, -3.0});
      get<1, 1>(t) = DataVector({-3.0, 5.0});
      get<1, 2>(t) = DataVector({4.0, 2.0});
      get<1, 3>(t) = DataVector({-5.0, 3.0});
      get<2, 0>(t) = DataVector({2.0, 3.0});
      get<2, 1>(t) = DataVector({-3.0, -5.0});
      get<2, 2>(t) = DataVector({4.0, 2.0});
      get<2, 3>(t) = DataVector({5.0, 3.0});
      get<3, 0>(t) = DataVector({2.0, 3.0});
      get<3, 1>(t) = DataVector({3.0, 5.0});
      get<3, 2>(t) = DataVector({-4.0, -2.0});
      get<3, 3>(t) = DataVector({5.0, 3.0});
      const auto det_inv = determinant_and_inverse<MyDetTag, MyInvTag>(t);
      CHECK_ITERABLE_APPROX(get(get<MyDetTag>(det_inv)),
                            DataVector({-960.0, 720.0}));
      CHECK_ITERABLE_APPROX((get<0, 0>(get<MyInvTag>(det_inv))),
                            DataVector({0.0, 1.0 / 6.0}));
      CHECK_ITERABLE_APPROX((get<0, 1>(get<MyInvTag>(det_inv))),
                            DataVector({0.25, -1.0 / 6.0}));
      CHECK_ITERABLE_APPROX((get<0, 2>(get<MyInvTag>(det_inv))),
                            DataVector({0.0, 0.0}));
      CHECK_ITERABLE_APPROX((get<0, 3>(get<MyInvTag>(det_inv))),
                            DataVector({0.25, 0.0}));
      CHECK_ITERABLE_APPROX((get<1, 0>(get<MyInvTag>(det_inv))),
                            DataVector({1.0 / 6.0, 0.1}));
      CHECK_ITERABLE_APPROX((get<1, 1>(get<MyInvTag>(det_inv))),
                            DataVector({0.0, 0.0}));
      CHECK_ITERABLE_APPROX((get<1, 2>(get<MyInvTag>(det_inv))),
                            DataVector({-1.0 / 6.0, -0.1}));
      CHECK_ITERABLE_APPROX((get<1, 3>(get<MyInvTag>(det_inv))),
                            DataVector({0.0, 0.0}));
      CHECK_ITERABLE_APPROX((get<2, 0>(get<MyInvTag>(det_inv))),
                            DataVector({0.125, 0.25}));
      CHECK_ITERABLE_APPROX((get<2, 1>(get<MyInvTag>(det_inv))),
                            DataVector({0.0, 0.0}));
      CHECK_ITERABLE_APPROX((get<2, 2>(get<MyInvTag>(det_inv))),
                            DataVector({0.0, 0.0}));
      CHECK_ITERABLE_APPROX((get<2, 3>(get<MyInvTag>(det_inv))),
                            DataVector({-0.125, -0.25}));
      CHECK_ITERABLE_APPROX((get<3, 0>(get<MyInvTag>(det_inv))),
                            DataVector({0.0, -1.0 / 6.0}));
      CHECK_ITERABLE_APPROX((get<3, 1>(get<MyInvTag>(det_inv))),
                            DataVector({-0.1, 1.0 / 6.0}));
      CHECK_ITERABLE_APPROX((get<3, 2>(get<MyInvTag>(det_inv))),
                            DataVector({0.1, 1.0 / 6.0}));
      CHECK_ITERABLE_APPROX((get<3, 3>(get<MyInvTag>(det_inv))),
                            DataVector({0.0, 1.0 / 6.0}));
    }
  }

  // Accuracy tests for ill-conditioned matrices
  test_4x4_near_singular_top_left_block();
  test_3x3_hilbert();
  test_3x3_large_eigenvalue_spread();
  test_3x3_spd_equilibration();
  test_refined_generic_3x3();
  test_refined_datavector_3x3();
}
