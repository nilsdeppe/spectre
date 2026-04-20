// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function computing the determinant and inverse of a tensor.

#pragma once

#include <cmath>
#include <type_traits>
#include <utility>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup TensorGroup
 * \brief Selects the algorithm used by `determinant_and_inverse`.
 */
enum class InversionMethod : uint8_t {
  /*!
   * \brief Direct Cramer's rule (cofactor expansion) for all dimensions.
   *
   * Error bound is $\mathcal{O}(n * \kappa(A) * u)$ uniformly, where $kappa(A)$
   * is the condition number of the full matrix and $u$ is unit roundoff. This
   * is the default and is suitable for well-conditioned matrices of any
   * structure.
   */
  Analytic,
  /*!
   * \brief For 4x4 tensors, uses partitioned 2x2-block inversion via
   * the Schur complement of the **top-left 2x2 block P** (rows/cols 0–1).
   *
   * Approximately 2.5x fewer flops than `Analytic` for 4x4. For dimensions
   * 1–3, identical to `Analytic`.
   *
   * \warning **P (top-left 2x2 block) must be well-conditioned.** If
   * $\kappa(P) \gg \kappa(A)$, the algorithm divides by $\det(P)$ early,
   * amplifying roundoff in every subsequent computation. The effective error
   * bound degrades to $\sim\kappa(P)^2 * u$, potentially losing many digits
   * even when the full matrix is well-conditioned.
   */
  BlockTopLeft,
  /*!
   * \brief For 4x4 tensors, uses partitioned 2x2-block inversion
   * via the Schur complement of the **bottom-right 2x2 block S** (rows/cols
   * 2–3).
   *
   * Same flop count as `BlockTopLeft`. For dimensions 1–3, identical to
   * `Analytic`.
   *
   * \warning **S (bottom-right 2x2 block) must be well-conditioned.** If
   * `kappa(S) >> kappa(A)`, the algorithm divides by `det(S)` early, causing
   * the same accuracy degradation as `BlockTopLeft` with an ill-conditioned P.
   */
  BlockBottomRight,
  /*!
   * \brief One Newton-Schulz refinement step $X_1 = X_0 + X_0 (I - A X_0)$
   * after an initial Cramer solve.
   *
   * The determinant is recomputed as $1 / det(X_1)$. Error bound drops to
   * $\mathcal{O}(\kappa(A)^2 * u^2)$, near machine precision for
   * $\kappa(A) < 10^8$. Roughly 3–4x more flops than `Analytic`.
   *
   * **Equilibration** (diagonal scaling \f$D_i = 1/\sqrt{A_{ii}}\f$ to reduce
   * magnitude disparity) is additionally applied before the Cramer solve for
   * **3×3 symmetric (SPD) tensors** (`Symmetry<1,1>` with `dim==3`). This is
   * the most common use case (spatial metric in GR) and achieves near machine
   * precision even when diagonal entries span many orders of magnitude.
   */
  Refined
};

namespace determinant_and_inverse_detail {
// Helps to shorten some repeated code:
template <typename Index0, typename Index1>
using inverse_indices =
    tmpl::list<change_index_up_lo<Index1>, change_index_up_lo<Index0>>;

template <typename Symm, typename Index0, typename Index1,
          typename = std::nullptr_t>
struct DetAndInverseImpl;

template <typename Symm, typename Index0, typename Index1>
struct DetAndInverseImpl<Symm, Index0, Index1, Requires<Index0::dim == 1>> {
  template <InversionMethod Method, typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<Tensor<T, Symm, inverse_indices<Index0, Index1>>*>
          inv,
      const Tensor<T, Symm, tmpl::list<Index0, Index1>>& tensor) {
    // Refined not yet implemented for 1D; falls through to Analytic.
    // BlockTopLeft/BlockBottomRight are identical to Analytic for dim < 4.
    const T& t00 = get<0, 0>(tensor);
    get(*det) = t00;
    get<0, 0>(*inv) = 1.0 / t00;
  }
};

// The inverse of a 2x2 tensor is computed from Cramer's rule.
template <typename Index0, typename Index1>
struct DetAndInverseImpl<Symmetry<2, 1>, Index0, Index1,
                         Requires<Index0::dim == 2>> {
  template <InversionMethod Method, typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<2, 1>, inverse_indices<Index0, Index1>>*>
          inv,
      const Tensor<T, Symmetry<2, 1>, tmpl::list<Index0, Index1>>& tensor) {
    // BlockTopLeft/BlockBottomRight are identical to Analytic for dim < 4.
    const T& t00 = get<0, 0>(tensor);
    const T& t01 = get<0, 1>(tensor);
    const T& t10 = get<1, 0>(tensor);
    const T& t11 = get<1, 1>(tensor);
    get(*det) = t00 * t11 - t01 * t10;
    if constexpr (Method == InversionMethod::Refined) {
      // 8 slots: R (4) in TempIj<0,2>; X_0 (4) in TempIj<1,2>.
      // X_0 written directly to buffer (no *inv round-trip).
      TempBuffer<tmpl::list<Tags::TempIj<0, 2, Frame::Inertial, T>,
                            Tags::TempIj<1, 2, Frame::Inertial, T>>>
          buffer(get_size(get<0, 0>(tensor)));
      auto& R = get<Tags::TempIj<0, 2, Frame::Inertial, T>>(buffer);
      auto& X = get<Tags::TempIj<1, 2, Frame::Inertial, T>>(buffer);
      auto& r00 = get<0, 0>(R);
      auto& r01 = get<0, 1>(R);
      auto& r10 = get<1, 0>(R);
      auto& r11 = get<1, 1>(R);
      auto& x00 = get<0, 0>(X);
      auto& x01 = get<0, 1>(X);
      auto& x10 = get<1, 0>(X);
      auto& x11 = get<1, 1>(X);
      // Phase 1: Cramer → X_0 directly in buffer (inverse_det-in-x00)
      x00 = 1.0 / get(*det);
      x01 = -t01 * x00;
      x10 = -t10 * x00;
      x11 = t00 * x00;
      x00 *= t11;
      // Phase 2: R = I - A*X_0 (fused, reading X_0 from buffer)
      r00 = 1.0 - (t00 * x00 + t01 * x10);
      r01 = -(t00 * x01 + t01 * x11);
      r10 = -(t10 * x00 + t11 * x10);
      r11 = 1.0 - (t10 * x01 + t11 * x11);
      // Phase 3: Y = X + X*R → *inv (no aliasing with buffer)
      get<0, 0>(*inv) = x00 + x00 * r00 + x01 * r10;
      get<0, 1>(*inv) = x01 + x00 * r01 + x01 * r11;
      get<1, 0>(*inv) = x10 + x10 * r00 + x11 * r10;
      get<1, 1>(*inv) = x11 + x10 * r01 + x11 * r11;
      // Phase 4: det = 1/det(Y)
      get(*det) = 1.0 / (get<0, 0>(*inv) * get<1, 1>(*inv) -
                         get<0, 1>(*inv) * get<1, 0>(*inv));
    } else {
      // Analytic / BlockTopLeft / BlockBottomRight: inverse_det-in-inv, no
      // buffer
      get<0, 0>(*inv) = 1.0 / get(*det);
      get<0, 1>(*inv) = -t01 * get<0, 0>(*inv);
      get<1, 0>(*inv) = -t10 * get<0, 0>(*inv);
      get<1, 1>(*inv) = t00 * get<0, 0>(*inv);
      get<0, 0>(*inv) *= t11;
    }
  }
};

template <typename Index0>
struct DetAndInverseImpl<Symmetry<1, 1>, Index0, Index0,
                         Requires<Index0::dim == 2>> {
  template <InversionMethod Method, typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<1, 1>, inverse_indices<Index0, Index0>>*>
          inv,
      const Tensor<T, Symmetry<1, 1>, tmpl::list<Index0, Index0>>& tensor) {
    const T& t00 = get<0, 0>(tensor);
    const T& t01 = get<0, 1>(tensor);
    const T& t11 = get<1, 1>(tensor);
    get(*det) = t00 * t11 - t01 * t01;
    if constexpr (Method == InversionMethod::Refined) {
      // 7 slots: R (4) in TempIj<0,2>; X_0 (3 sym) in Tempij<0,2>.
      // X_0 written directly to buffer (no *inv round-trip).
      TempBuffer<tmpl::list<Tags::TempIj<0, 2, Frame::Inertial, T>,
                            Tags::Tempij<0, 2, Frame::Inertial, T>>>
          buffer(get_size(get<0, 0>(tensor)));
      auto& R = get<Tags::TempIj<0, 2, Frame::Inertial, T>>(buffer);
      auto& X = get<Tags::Tempij<0, 2, Frame::Inertial, T>>(buffer);
      auto& r00 = get<0, 0>(R);
      auto& r01 = get<0, 1>(R);
      auto& r10 = get<1, 0>(R);
      auto& r11 = get<1, 1>(R);
      auto& x00 = get<0, 0>(X);
      auto& x01 = get<0, 1>(X);
      auto& x11 = get<1, 1>(X);
      // Phase 1: Cramer → X_0 directly in buffer (inverse_det-in-x00: x00 =
      // 1/det, used for x01,x11, then x00 *= t11 to get cofactor(0,0)/det)
      x00 = 1.0 / get(*det);
      x01 = -t01 * x00;
      x11 = t00 * x00;
      x00 *= t11;
      // Phase 2: R = I - A*X_0 (fused; A sym: t10=t01; X_0 sym: x10=x01)
      r00 = 1.0 - (t00 * x00 + t01 * x01);
      r01 = -(t00 * x01 + t01 * x11);
      r10 = -(t01 * x00 + t11 * x01);
      r11 = 1.0 - (t01 * x01 + t11 * x11);
      // Phase 3: Y = X + sym(X*R) → *inv (no aliasing with buffer)
      // Y_ij = x_ij + 0.5*(C_ij + C_ji), C = X*R
      get<0, 0>(*inv) = x00 + x00 * r00 + x01 * r10;
      get<0, 1>(*inv) =
          x01 + 0.5 * (x00 * r01 + x01 * r11 + x01 * r00 + x11 * r10);
      get<1, 1>(*inv) = x11 + x01 * r01 + x11 * r11;
      // Phase 4: det = 1/det(Y)
      get(*det) = 1.0 / (get<0, 0>(*inv) * get<1, 1>(*inv) -
                         get<0, 1>(*inv) * get<0, 1>(*inv));
    } else {
      // Analytic: inverse_det-in-inv, no buffer
      get<0, 0>(*inv) = 1.0 / get(*det);
      get<0, 1>(*inv) = -t01 * get<0, 0>(*inv);
      get<1, 1>(*inv) = t00 * get<0, 0>(*inv);
      get<0, 0>(*inv) *= t11;
    }
  }
};

// The inverse of a 3x3 tensor is computed from Cramer's rule. By reusing some
// terms, the determinant is computed efficiently at the same time.
template <typename Index0, typename Index1>
struct DetAndInverseImpl<Symmetry<2, 1>, Index0, Index1,
                         Requires<Index0::dim == 3>> {
  template <InversionMethod Method, typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<2, 1>, inverse_indices<Index0, Index1>>*>
          inv,
      const Tensor<T, Symmetry<2, 1>, tmpl::list<Index0, Index1>>& tensor) {
    const T& t00 = get<0, 0>(tensor);
    const T& t01 = get<0, 1>(tensor);
    const T& t02 = get<0, 2>(tensor);
    const T& t10 = get<1, 0>(tensor);
    const T& t11 = get<1, 1>(tensor);
    const T& t12 = get<1, 2>(tensor);
    const T& t20 = get<2, 0>(tensor);
    const T& t21 = get<2, 1>(tensor);
    const T& t22 = get<2, 2>(tensor);
    if constexpr (Method == InversionMethod::Refined) {
      // 18 slots: R (9) in TempIj<0,3>; X_0 (9) in TempIj<1,3>.
      // X_0 written directly to buffer (no *inv round-trip).
      TempBuffer<tmpl::list<Tags::TempIj<0, 3, Frame::Inertial, T>,
                            Tags::TempIj<1, 3, Frame::Inertial, T>>>
          buffer(get_size(get<0, 0>(tensor)));
      auto& R = get<Tags::TempIj<0, 3, Frame::Inertial, T>>(buffer);
      auto& X = get<Tags::TempIj<1, 3, Frame::Inertial, T>>(buffer);
      auto& r00 = get<0, 0>(R);
      auto& r01 = get<0, 1>(R);
      auto& r02 = get<0, 2>(R);
      auto& r10 = get<1, 0>(R);
      auto& r11 = get<1, 1>(R);
      auto& r12 = get<1, 2>(R);
      auto& r20 = get<2, 0>(R);
      auto& r21 = get<2, 1>(R);
      auto& r22 = get<2, 2>(R);
      auto& x00 = get<0, 0>(X);
      auto& x01 = get<0, 1>(X);
      auto& x02 = get<0, 2>(X);
      auto& x10 = get<1, 0>(X);
      auto& x11 = get<1, 1>(X);
      auto& x12 = get<1, 2>(X);
      auto& x20 = get<2, 0>(X);
      auto& x21 = get<2, 1>(X);
      auto& x22 = get<2, 2>(X);
      // Phase 1: Cramer → X_0 directly in buffer (inverse_det-in-x01: x01 =
      // 1/det, used for all entries, then x01 *= cofactor(0,1) last)
      x00 = t11 * t22 - t12 * t21;  // cofactor(0,0) temporarily
      x10 = t12 * t20 - t10 * t22;  // cofactor(1,0) temporarily
      x20 = t10 * t21 - t11 * t20;  // cofactor(2,0) temporarily
      get(*det) = t00 * x00 + t01 * x10 + t02 * x20;
      x01 = 1.0 / get(*det);  // inverse_det in x01; set last
      x00 *= x01;
      x10 *= x01;
      x20 *= x01;
      x02 = (t01 * t12 - t02 * t11) * x01;
      x11 = (t22 * t00 - t20 * t02) * x01;
      x12 = (t02 * t10 - t00 * t12) * x01;
      x21 = (t20 * t01 - t21 * t00) * x01;
      x22 = (t00 * t11 - t01 * t10) * x01;
      x01 *= (t21 * t02 - t22 * t01);  // cofactor(0,1)/det — set LAST
      // Phase 2: R = I - A*X_0 (fused; reads X_0 from buffer)
      r00 = 1.0 - (t00 * x00 + t01 * x10 + t02 * x20);
      r01 = -(t00 * x01 + t01 * x11 + t02 * x21);
      r02 = -(t00 * x02 + t01 * x12 + t02 * x22);
      r10 = -(t10 * x00 + t11 * x10 + t12 * x20);
      r11 = 1.0 - (t10 * x01 + t11 * x11 + t12 * x21);
      r12 = -(t10 * x02 + t11 * x12 + t12 * x22);
      r20 = -(t20 * x00 + t21 * x10 + t22 * x20);
      r21 = -(t20 * x01 + t21 * x11 + t22 * x21);
      r22 = 1.0 - (t20 * x02 + t21 * x12 + t22 * x22);
      // Phase 3: Y = X + X*R → *inv (no aliasing with buffer)
      get<0, 0>(*inv) = x00 + x00 * r00 + x01 * r10 + x02 * r20;
      get<0, 1>(*inv) = x01 + x00 * r01 + x01 * r11 + x02 * r21;
      get<0, 2>(*inv) = x02 + x00 * r02 + x01 * r12 + x02 * r22;
      get<1, 0>(*inv) = x10 + x10 * r00 + x11 * r10 + x12 * r20;
      get<1, 1>(*inv) = x11 + x10 * r01 + x11 * r11 + x12 * r21;
      get<1, 2>(*inv) = x12 + x10 * r02 + x11 * r12 + x12 * r22;
      get<2, 0>(*inv) = x20 + x20 * r00 + x21 * r10 + x22 * r20;
      get<2, 1>(*inv) = x21 + x20 * r01 + x21 * r11 + x22 * r21;
      get<2, 2>(*inv) = x22 + x20 * r02 + x21 * r12 + x22 * r22;
      // Phase 4: det = 1/det(Y) from *inv
      const T& y00 = get<0, 0>(*inv);
      const T& y01 = get<0, 1>(*inv);
      const T& y02 = get<0, 2>(*inv);
      const T& y10 = get<1, 0>(*inv);
      const T& y11 = get<1, 1>(*inv);
      const T& y12 = get<1, 2>(*inv);
      const T& y20 = get<2, 0>(*inv);
      const T& y21 = get<2, 1>(*inv);
      const T& y22 = get<2, 2>(*inv);
      get(*det) =
          1.0 / (y00 * (y11 * y22 - y12 * y21) - y01 * (y10 * y22 - y12 * y20) +
                 y02 * (y10 * y21 - y11 * y20));
    } else {
      // Analytic / Block: cofactors-in-inv + inverse_det-in-inv, no buffer
      get<0, 0>(*inv) = t11 * t22 - t12 * t21;  // C[0,0]
      get<1, 0>(*inv) = t12 * t20 - t10 * t22;  // C[0,1]
      get<2, 0>(*inv) = t10 * t21 - t11 * t20;  // C[0,2]
      get(*det) =
          t00 * get<0, 0>(*inv) + t01 * get<1, 0>(*inv) + t02 * get<2, 0>(*inv);
      get<0, 1>(*inv) = 1.0 / get(*det);  // inverse_det; overwritten last
      get<0, 0>(*inv) *= get<0, 1>(*inv);
      get<1, 0>(*inv) *= get<0, 1>(*inv);
      get<2, 0>(*inv) *= get<0, 1>(*inv);
      get<0, 2>(*inv) = (t01 * t12 - t02 * t11) * get<0, 1>(*inv);
      get<1, 1>(*inv) = (t22 * t00 - t20 * t02) * get<0, 1>(*inv);
      get<1, 2>(*inv) = (t02 * t10 - t00 * t12) * get<0, 1>(*inv);
      get<2, 1>(*inv) = (t20 * t01 - t21 * t00) * get<0, 1>(*inv);
      get<2, 2>(*inv) = (t00 * t11 - t01 * t10) * get<0, 1>(*inv);
      get<0, 1>(*inv) *= (t21 * t02 - t22 * t01);  // C[1,0]/det — set LAST
    }
  }
};

template <typename Index0>
struct DetAndInverseImpl<Symmetry<1, 1>, Index0, Index0,
                         Requires<Index0::dim == 3>> {
  template <InversionMethod Method, typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<1, 1>, inverse_indices<Index0, Index0>>*>
          inv,
      const Tensor<T, Symmetry<1, 1>, tmpl::list<Index0, Index0>>& tensor) {
    const T& t00 = get<0, 0>(tensor);
    const T& t01 = get<0, 1>(tensor);
    const T& t02 = get<0, 2>(tensor);
    const T& t11 = get<1, 1>(tensor);
    const T& t12 = get<1, 2>(tensor);
    const T& t22 = get<2, 2>(tensor);
    if constexpr (Method == InversionMethod::Refined) {
      // 15 slots with aggressive reuse across phases (single allocation):
      //   Slots 0-2:   d0,d1,d2  [Ph1-3] → r00-r02  [Ph4]
      //   Slots 3-5:   ts01,ts02,ts12  [Ph1-2] → x00,x01,x02  [Ph3]
      //                              → r10-r12  [Ph4]
      //   Slots 6-8:   as,bs,cs  [Ph2] → x11,x12,x22  [Ph3]
      //                              → r20-r22  [Ph4]
      //   Slot  9:     det_scaled/inverse_det_s  [Ph2]  (in-place 1/x)
      //   Slots 10-14: xs00-xs12  [Ph2-3]
      TempBuffer<tmpl::list<Tags::TempScalar<0, T>, Tags::TempScalar<1, T>,
                            Tags::TempScalar<2, T>, Tags::TempScalar<3, T>,
                            Tags::TempScalar<4, T>, Tags::TempScalar<5, T>,
                            Tags::TempScalar<6, T>, Tags::TempScalar<7, T>,
                            Tags::TempScalar<8, T>, Tags::TempScalar<9, T>,
                            Tags::TempScalar<10, T>, Tags::TempScalar<11, T>,
                            Tags::TempScalar<12, T>, Tags::TempScalar<13, T>,
                            Tags::TempScalar<14, T>>>
          buffer(get_size(get<0, 0>(tensor)));
      // Phase 1: Equilibration scaling factors
      // Assumes diagonal entries > 0 (SPD or similar)
      using std::sqrt;
      auto& d0 = get(get<Tags::TempScalar<0, T>>(buffer));
      auto& d1 = get(get<Tags::TempScalar<1, T>>(buffer));
      auto& d2 = get(get<Tags::TempScalar<2, T>>(buffer));
      d0 = 1.0 / sqrt(t00);
      d1 = 1.0 / sqrt(t11);
      d2 = 1.0 / sqrt(t22);
      auto& ts01 = get(get<Tags::TempScalar<3, T>>(buffer));
      auto& ts02 = get(get<Tags::TempScalar<4, T>>(buffer));
      auto& ts12 = get(get<Tags::TempScalar<5, T>>(buffer));
      ts01 = t01 * d0 * d1;
      ts02 = t02 * d0 * d2;
      ts12 = t12 * d1 * d2;
      // Phase 2: Cramer's rule on scaled matrix (diagonal = 1)
      auto& as = get(get<Tags::TempScalar<6, T>>(buffer));
      auto& bs = get(get<Tags::TempScalar<7, T>>(buffer));
      auto& cs = get(get<Tags::TempScalar<8, T>>(buffer));
      as = 1.0 - ts12 * ts12;
      bs = ts12 * ts02 - ts01;
      cs = ts01 * ts12 - ts02;
      auto& inverse_det_s = get(get<Tags::TempScalar<9, T>>(buffer));
      inverse_det_s =
          as + ts01 * bs + ts02 * cs;  // det_scaled, then in-place 1/x
      inverse_det_s = 1.0 / inverse_det_s;
      auto& xs00 = get(get<Tags::TempScalar<10, T>>(buffer));
      auto& xs01 = get(get<Tags::TempScalar<11, T>>(buffer));
      auto& xs02 = get(get<Tags::TempScalar<12, T>>(buffer));
      auto& xs11 = get(get<Tags::TempScalar<13, T>>(buffer));
      auto& xs12 = get(get<Tags::TempScalar<14, T>>(buffer));
      xs00 = as * inverse_det_s;
      xs01 = bs * inverse_det_s;
      xs02 = cs * inverse_det_s;
      xs11 = (1.0 - ts02 * ts02) * inverse_det_s;
      xs12 = (ts02 * ts01 - ts12) * inverse_det_s;
      // ts01,ts02,ts12,as,bs,cs,inverse_det_s all scope end; d0,d1,d2,xs alive
      // Phase 3: Unscale into slots 3-8 (ts/as/bs/cs end of scope).
      // x22 computed first (inlines xs22 = 1-ts01^2)*inverse_det_s before ts01
      // slot is overwritten by x00).
      auto& x00 = get(get<Tags::TempScalar<3, T>>(buffer));  // reuse ts01
      auto& x01 = get(get<Tags::TempScalar<4, T>>(buffer));  // reuse ts02
      auto& x02 = get(get<Tags::TempScalar<5, T>>(buffer));  // reuse ts12
      auto& x11 = get(get<Tags::TempScalar<6, T>>(buffer));  // reuse as
      auto& x12 = get(get<Tags::TempScalar<7, T>>(buffer));  // reuse bs
      auto& x22 = get(get<Tags::TempScalar<8, T>>(buffer));  // reuse cs
      x22 = (1.0 - ts01 * ts01) * inverse_det_s * d2 *
            d2;  // ts01,inverse_det_s,d2 still alive
      x00 = xs00 * d0 * d0;
      x01 = xs01 * d0 * d1;
      x02 = xs02 * d0 * d2;  // d0 scope ends
      x11 = xs11 * d1 * d1;
      x12 = xs12 * d1 * d2;  // d1,d2 scope ends
      // d0,d1,d2 scope end; xs00-xs12 scope end; x00-x22 alive in slots 3-8
      // Phase 4: R = I-AX (fused) into slots 0-8; Y = X + sym(X*R) into *inv
      auto& r00 = get(get<Tags::TempScalar<0, T>>(buffer));  // reuse d0
      auto& r01 = get(get<Tags::TempScalar<1, T>>(buffer));  // reuse d1
      auto& r02 = get(get<Tags::TempScalar<2, T>>(buffer));  // reuse d2
      auto& r10 =
          get(get<Tags::TempScalar<9, T>>(buffer));  // reuse inverse_det_s
      auto& r11 = get(get<Tags::TempScalar<10, T>>(buffer));  // reuse xs00
      auto& r12 = get(get<Tags::TempScalar<11, T>>(buffer));  // reuse xs01
      auto& r20 = get(get<Tags::TempScalar<12, T>>(buffer));  // reuse xs02
      auto& r21 = get(get<Tags::TempScalar<13, T>>(buffer));  // reuse xs11
      auto& r22 = get(get<Tags::TempScalar<14, T>>(buffer));  // reuse xs12
      r00 = 1.0 - (t00 * x00 + t01 * x01 + t02 * x02);
      r01 = -(t00 * x01 + t01 * x11 + t02 * x12);
      r02 = -(t00 * x02 + t01 * x12 + t02 * x22);
      r10 = -(t01 * x00 + t11 * x01 + t12 * x02);
      r11 = 1.0 - (t01 * x01 + t11 * x11 + t12 * x12);
      r12 = -(t01 * x02 + t11 * x12 + t12 * x22);
      r20 = -(t02 * x00 + t12 * x01 + t22 * x02);
      r21 = -(t02 * x01 + t12 * x11 + t22 * x12);
      r22 = 1.0 - (t02 * x02 + t12 * x12 + t22 * x22);
      // Y = X + (X*R + (X*R)^T)/2 written directly to *inv.
      // Diagonal: Y_ii = x_ii + C_ii (C = X*R; C^T_ii = C_ii for sym)
      get<0, 0>(*inv) = x00 + (x00 * r00 + x01 * r10 + x02 * r20);  // Y_00
      get<1, 1>(*inv) = x11 + (x01 * r01 + x11 * r11 + x12 * r21);  // Y_11
      get<2, 2>(*inv) = x22 + (x02 * r02 + x12 * r12 + x22 * r22);  // Y_22
      // Off-diagonal: inline C pair; Y_ij = x_ij + 0.5*(C_ij + C_ji)
      get<0, 1>(*inv) = x01 + 0.5 * (x00 * r01 + x01 * r11 + x02 * r21 +
                                     x01 * r00 + x11 * r10 + x12 * r20);
      get<0, 2>(*inv) = x02 + 0.5 * (x00 * r02 + x01 * r12 + x02 * r22 +
                                     x02 * r00 + x12 * r10 + x22 * r20);
      get<1, 2>(*inv) = x12 + 0.5 * (x01 * r02 + x11 * r12 + x12 * r22 +
                                     x02 * r01 + x12 * r11 + x22 * r21);
      // Phase 5: det = 1/det(Y); read Y from *inv directly
      const T& y00 = get<0, 0>(*inv);
      const T& y01 = get<0, 1>(*inv);
      const T& y02 = get<0, 2>(*inv);
      const T& y11 = get<1, 1>(*inv);
      const T& y12 = get<1, 2>(*inv);
      const T& y22 = get<2, 2>(*inv);
      get(*det) =
          1.0 / (y00 * (y11 * y22 - y12 * y12) - y01 * (y01 * y22 - y12 * y02) +
                 y02 * (y01 * y12 - y11 * y02));
    } else {
      // Analytic: cofactors-in-inv + inverse_det-in-inv, no buffer
      get<0, 0>(*inv) = t11 * t22 - t12 * t12;  // C[0,0]
      get<0, 1>(*inv) = t12 * t02 - t01 * t22;  // C[0,1]
      get<0, 2>(*inv) = t01 * t12 - t11 * t02;  // C[0,2]
      get(*det) =
          t00 * get<0, 0>(*inv) + t01 * get<0, 1>(*inv) + t02 * get<0, 2>(*inv);
      get<1, 1>(*inv) = 1.0 / get(*det);  // inverse_det; overwritten last
      get<0, 0>(*inv) *= get<1, 1>(*inv);
      get<0, 1>(*inv) *= get<1, 1>(*inv);
      get<0, 2>(*inv) *= get<1, 1>(*inv);
      get<1, 2>(*inv) = (t02 * t01 - t00 * t12) * get<1, 1>(*inv);
      get<2, 2>(*inv) = (t00 * t11 - t01 * t01) * get<1, 1>(*inv);
      get<1, 1>(*inv) *= (t22 * t00 - t02 * t02);  // C[1,1]/det — set LAST
    }
  }
};

// 4x4 generic (Symmetry<2,1>):
//
// - BlockTopLeft: existing partitioned 2x2-block inversion via Schur
//   complement of P (top-left block). Fast but accuracy degrades when P is
//   ill-conditioned relative to the full matrix.
//
// - BlockBottomRight: partitioned 2x2-block inversion via Schur complement of
//   S (bottom-right block). Same flop count as BlockTopLeft but accuracy
//   degrades when S is ill-conditioned relative to the full matrix.
//
// - Analytic / Refined: direct Cramer's rule via cofactor expansion. Uses 18
//   precomputed 2x2 minors (6 each from row-pairs {2,3}, {1,3}, {1,2}).
//   Error bound is uniformly O(4 * kappa(A) * u) with no dependence on
//   sub-block conditioning.
template <typename Index0, typename Index1>
struct DetAndInverseImpl<Symmetry<2, 1>, Index0, Index1,
                         Requires<Index0::dim == 4>> {
  template <InversionMethod Method, typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<2, 1>, inverse_indices<Index0, Index1>>*>
          inv,
      const Tensor<T, Symmetry<2, 1>, tmpl::list<Index0, Index1>>& tensor) {
    if constexpr (Method == InversionMethod::BlockTopLeft) {
      // Partitioned block inversion using the Schur complement of P
      // (top-left 2x2 block). See the InversionMethod documentation for the
      // accuracy tradeoffs.
      const T& p00 = get<0, 0>(tensor);
      const T& p01 = get<0, 1>(tensor);
      const T& p10 = get<1, 0>(tensor);
      const T& p11 = get<1, 1>(tensor);
      const T& q00 = get<0, 2>(tensor);
      const T& q01 = get<0, 3>(tensor);
      const T& q10 = get<1, 2>(tensor);
      const T& q11 = get<1, 3>(tensor);
      const T& r00 = get<2, 0>(tensor);
      const T& r01 = get<2, 1>(tensor);
      const T& r10 = get<3, 0>(tensor);
      const T& r11 = get<3, 1>(tensor);
      const T& s00 = get<2, 2>(tensor);
      const T& s01 = get<2, 3>(tensor);
      const T& s10 = get<3, 2>(tensor);
      const T& s11 = get<3, 3>(tensor);

      T& u00 = get<0, 0>(*inv);
      T& u01 = get<0, 1>(*inv);
      T& u10 = get<1, 0>(*inv);
      T& u11 = get<1, 1>(*inv);
      T& v00 = get<0, 2>(*inv);
      T& v01 = get<0, 3>(*inv);
      T& v10 = get<1, 2>(*inv);
      T& v11 = get<1, 3>(*inv);
      T& w00 = get<2, 0>(*inv);
      T& w01 = get<2, 1>(*inv);
      T& w10 = get<3, 0>(*inv);
      T& w11 = get<3, 1>(*inv);
      T& x00 = get<2, 2>(*inv);
      T& x01 = get<2, 3>(*inv);
      T& x10 = get<3, 2>(*inv);
      T& x11 = get<3, 3>(*inv);

      // Temporarily store det(P) in det
      get(*det) = p00 * p11 - p01 * p10;
      // 12 slots: inv_p (4), r_inv_p (4), inv_p_q (4).
      // inverse_det_p in u00 (overwritten later); Schur complement in x block;
      // x inversion uses u00, u01 as scratch (both overwritten in final step).
      TempBuffer<tmpl::list<Tags::TempIj<0, 2, Frame::Inertial, T>,
                            Tags::TempIj<1, 2, Frame::Inertial, T>,
                            Tags::TempIj<2, 2, Frame::Inertial, T>>>
          buffer(get_size(get<0, 0>(tensor)));
      u00 = 1.0 / get(*det);  // inverse_det_p; overwritten later
      auto& inv_p = get<Tags::TempIj<0, 2, Frame::Inertial, T>>(buffer);
      auto& inv_p00 = get<0, 0>(inv_p);
      auto& inv_p01 = get<0, 1>(inv_p);
      auto& inv_p10 = get<1, 0>(inv_p);
      auto& inv_p11 = get<1, 1>(inv_p);
      inv_p00 = p11 * u00;
      inv_p01 = -p01 * u00;
      inv_p10 = -p10 * u00;
      inv_p11 = p00 * u00;
      auto& r_inv_p = get<Tags::TempIj<1, 2, Frame::Inertial, T>>(buffer);
      auto& r_inv_p00 = get<0, 0>(r_inv_p);
      auto& r_inv_p01 = get<0, 1>(r_inv_p);
      auto& r_inv_p10 = get<1, 0>(r_inv_p);
      auto& r_inv_p11 = get<1, 1>(r_inv_p);
      r_inv_p00 = r00 * inv_p00 + r01 * inv_p10;
      r_inv_p01 = r00 * inv_p01 + r01 * inv_p11;
      r_inv_p10 = r10 * inv_p00 + r11 * inv_p10;
      r_inv_p11 = r10 * inv_p01 + r11 * inv_p11;
      auto& inv_p_q = get<Tags::TempIj<2, 2, Frame::Inertial, T>>(buffer);
      auto& inv_p_q00 = get<0, 0>(inv_p_q);
      auto& inv_p_q01 = get<0, 1>(inv_p_q);
      auto& inv_p_q10 = get<1, 0>(inv_p_q);
      auto& inv_p_q11 = get<1, 1>(inv_p_q);
      inv_p_q00 = inv_p00 * q00 + inv_p01 * q10;
      inv_p_q01 = inv_p00 * q01 + inv_p01 * q11;
      inv_p_q10 = inv_p10 * q00 + inv_p11 * q10;
      inv_p_q11 = inv_p10 * q01 + inv_p11 * q11;
      // Schur complement → x block directly
      x00 = s00 - (r_inv_p00 * q00 + r_inv_p01 * q10);
      x01 = s01 - (r_inv_p00 * q01 + r_inv_p01 * q11);
      x10 = s10 - (r_inv_p10 * q00 + r_inv_p11 * q10);
      x11 = s11 - (r_inv_p10 * q01 + r_inv_p11 * q11);
      // Invert x in-place: save x00 in u00, det(x) in u01 (both overwritten
      // in the final u-block step below)
      u00 = x00;
      u01 = x00 * x11 - x01 * x10;
      get(*det) *= u01;
      u01 = 1.0 / u01;
      x00 = x11 * u01;
      x11 = u00 * u01;
      x01 = -x01 * u01;
      x10 = -x10 * u01;
      w00 = -x00 * r_inv_p00 - x01 * r_inv_p10;
      w01 = -x00 * r_inv_p01 - x01 * r_inv_p11;
      w10 = -x10 * r_inv_p00 - x11 * r_inv_p10;
      w11 = -x10 * r_inv_p01 - x11 * r_inv_p11;
      v00 = -inv_p_q00 * x00 - inv_p_q01 * x10;
      v01 = -inv_p_q00 * x01 - inv_p_q01 * x11;
      v10 = -inv_p_q10 * x00 - inv_p_q11 * x10;
      v11 = -inv_p_q10 * x01 - inv_p_q11 * x11;
      u00 = inv_p00 - (inv_p_q00 * w00 + inv_p_q01 * w10);
      u01 = inv_p01 - (inv_p_q00 * w01 + inv_p_q01 * w11);
      u10 = inv_p10 - (inv_p_q10 * w00 + inv_p_q11 * w10);
      u11 = inv_p11 - (inv_p_q10 * w01 + inv_p_q11 * w11);

    } else if constexpr (Method == InversionMethod::BlockBottomRight) {
      // Partitioned block inversion using the Schur complement of S
      // (bottom-right 2x2 block). See the InversionMethod documentation for
      // the accuracy tradeoffs.
      //
      // Algorithm:
      //   inv(S):           inv_s
      //   Q * inv(S):       q_inv_s
      //   inv(S) * R:       inv_s_r
      //   Schur(S):         P - Q * inv(S) * R  →  inv_u (before inversion)
      //   U = inv(Schur):   u block of output
      //   V = -U*Q*inv(S):  v block of output
      //   W = -inv(S)*R*U:  w block of output
      //   X = inv(S) - W*(Q*inv(S)):  x block of output
      //   det(A) = det(S) * det(Schur(S))
      const T& p00 = get<0, 0>(tensor);
      const T& p01 = get<0, 1>(tensor);
      const T& p10 = get<1, 0>(tensor);
      const T& p11 = get<1, 1>(tensor);
      const T& q00 = get<0, 2>(tensor);
      const T& q01 = get<0, 3>(tensor);
      const T& q10 = get<1, 2>(tensor);
      const T& q11 = get<1, 3>(tensor);
      const T& r00 = get<2, 0>(tensor);
      const T& r01 = get<2, 1>(tensor);
      const T& r10 = get<3, 0>(tensor);
      const T& r11 = get<3, 1>(tensor);
      const T& s00 = get<2, 2>(tensor);
      const T& s01 = get<2, 3>(tensor);
      const T& s10 = get<3, 2>(tensor);
      const T& s11 = get<3, 3>(tensor);

      T& u00 = get<0, 0>(*inv);
      T& u01 = get<0, 1>(*inv);
      T& u10 = get<1, 0>(*inv);
      T& u11 = get<1, 1>(*inv);
      T& v00 = get<0, 2>(*inv);
      T& v01 = get<0, 3>(*inv);
      T& v10 = get<1, 2>(*inv);
      T& v11 = get<1, 3>(*inv);
      T& w00 = get<2, 0>(*inv);
      T& w01 = get<2, 1>(*inv);
      T& w10 = get<3, 0>(*inv);
      T& w11 = get<3, 1>(*inv);
      T& x00 = get<2, 2>(*inv);
      T& x01 = get<2, 3>(*inv);
      T& x10 = get<3, 2>(*inv);
      T& x11 = get<3, 3>(*inv);

      // Temporarily store det(S) in det
      get(*det) = s00 * s11 - s01 * s10;
      // 12 slots: inv_s (4), q_inv_s (4), inv_s_r (4).
      // inverse_det_s in x00 (overwritten later); Schur complement in u block;
      // u inversion uses x00, x01 as scratch (both overwritten in final step).
      TempBuffer<tmpl::list<Tags::TempIj<0, 2, Frame::Inertial, T>,
                            Tags::TempIj<1, 2, Frame::Inertial, T>,
                            Tags::TempIj<2, 2, Frame::Inertial, T>>>
          buffer(get_size(get<0, 0>(tensor)));
      x00 = 1.0 / get(*det);  // inverse_det_s; overwritten later
      auto& inv_s = get<Tags::TempIj<0, 2, Frame::Inertial, T>>(buffer);
      auto& inv_s00 = get<0, 0>(inv_s);
      auto& inv_s01 = get<0, 1>(inv_s);
      auto& inv_s10 = get<1, 0>(inv_s);
      auto& inv_s11 = get<1, 1>(inv_s);
      inv_s00 = s11 * x00;
      inv_s01 = -s01 * x00;
      inv_s10 = -s10 * x00;
      inv_s11 = s00 * x00;
      // Q * inv(S)
      auto& q_inv_s = get<Tags::TempIj<1, 2, Frame::Inertial, T>>(buffer);
      auto& q_inv_s00 = get<0, 0>(q_inv_s);
      auto& q_inv_s01 = get<0, 1>(q_inv_s);
      auto& q_inv_s10 = get<1, 0>(q_inv_s);
      auto& q_inv_s11 = get<1, 1>(q_inv_s);
      q_inv_s00 = q00 * inv_s00 + q01 * inv_s10;
      q_inv_s01 = q00 * inv_s01 + q01 * inv_s11;
      q_inv_s10 = q10 * inv_s00 + q11 * inv_s10;
      q_inv_s11 = q10 * inv_s01 + q11 * inv_s11;
      // inv(S) * R
      auto& inv_s_r = get<Tags::TempIj<2, 2, Frame::Inertial, T>>(buffer);
      auto& inv_s_r00 = get<0, 0>(inv_s_r);
      auto& inv_s_r01 = get<0, 1>(inv_s_r);
      auto& inv_s_r10 = get<1, 0>(inv_s_r);
      auto& inv_s_r11 = get<1, 1>(inv_s_r);
      inv_s_r00 = inv_s00 * r00 + inv_s01 * r10;
      inv_s_r01 = inv_s00 * r01 + inv_s01 * r11;
      inv_s_r10 = inv_s10 * r00 + inv_s11 * r10;
      inv_s_r11 = inv_s10 * r01 + inv_s11 * r11;
      // Schur complement of S → u block directly
      u00 = p00 - (q_inv_s00 * r00 + q_inv_s01 * r10);
      u01 = p01 - (q_inv_s00 * r01 + q_inv_s01 * r11);
      u10 = p10 - (q_inv_s10 * r00 + q_inv_s11 * r10);
      u11 = p11 - (q_inv_s10 * r01 + q_inv_s11 * r11);
      // Invert u in-place: save u00 in x00, det(u) in x01 (both overwritten
      // in the final x-block step below)
      x00 = u00;
      x01 = u00 * u11 - u01 * u10;
      get(*det) *= x01;
      x01 = 1.0 / x01;
      u00 = u11 * x01;
      u11 = x00 * x01;
      u01 = -u01 * x01;
      u10 = -u10 * x01;
      // V = -U * Q * inv(S)
      v00 = -(u00 * q_inv_s00 + u01 * q_inv_s10);
      v01 = -(u00 * q_inv_s01 + u01 * q_inv_s11);
      v10 = -(u10 * q_inv_s00 + u11 * q_inv_s10);
      v11 = -(u10 * q_inv_s01 + u11 * q_inv_s11);
      // W = -inv(S) * R * U
      w00 = -(inv_s_r00 * u00 + inv_s_r01 * u10);
      w01 = -(inv_s_r00 * u01 + inv_s_r01 * u11);
      w10 = -(inv_s_r10 * u00 + inv_s_r11 * u10);
      w11 = -(inv_s_r10 * u01 + inv_s_r11 * u11);
      // X = inv(S) - W * (Q * inv(S))
      x00 = inv_s00 - (w00 * q_inv_s00 + w01 * q_inv_s10);
      x01 = inv_s01 - (w00 * q_inv_s01 + w01 * q_inv_s11);
      x10 = inv_s10 - (w10 * q_inv_s00 + w11 * q_inv_s10);
      x11 = inv_s11 - (w10 * q_inv_s01 + w11 * q_inv_s11);

    } else {
      // Analytic (and Refined, which adds a Newton-Schulz step after):
      // Direct Cramer's rule via cofactor expansion.
      //
      // Precompute 18 independent 2x2 minors from three row pairs.
      // Naming: m{rows}_{cols}, e.g. m23_13 = minor of rows 2,3 and cols 1,3
      //   = t_{2,1}*t_{3,3} - t_{2,3}*t_{3,1}
      const T& t00 = get<0, 0>(tensor);
      const T& t01 = get<0, 1>(tensor);
      const T& t02 = get<0, 2>(tensor);
      const T& t03 = get<0, 3>(tensor);
      const T& t10 = get<1, 0>(tensor);
      const T& t11 = get<1, 1>(tensor);
      const T& t12 = get<1, 2>(tensor);
      const T& t13 = get<1, 3>(tensor);
      const T& t20 = get<2, 0>(tensor);
      const T& t21 = get<2, 1>(tensor);
      const T& t22 = get<2, 2>(tensor);
      const T& t23 = get<2, 3>(tensor);
      const T& t30 = get<3, 0>(tensor);
      const T& t31 = get<3, 1>(tensor);
      const T& t32 = get<3, 2>(tensor);
      const T& t33 = get<3, 3>(tensor);
      if constexpr (Method == InversionMethod::Refined) {
        // 32 slots: R (16) in TempIj<0,4>; X_0 (16) in TempIj<1,4>.
        // X_0 written directly to buffer (no *inv round-trip).
        // Phase 1 reuses R[0,0..1,1] (6 slots) as minor temporaries per group.
        TempBuffer<tmpl::list<Tags::TempIj<0, 4, Frame::Inertial, T>,
                              Tags::TempIj<1, 4, Frame::Inertial, T>>>
            buffer(get_size(get<0, 0>(tensor)));
        auto& R = get<Tags::TempIj<0, 4, Frame::Inertial, T>>(buffer);
        auto& X = get<Tags::TempIj<1, 4, Frame::Inertial, T>>(buffer);
        auto& x00 = get<0, 0>(X);
        auto& x01 = get<0, 1>(X);
        auto& x02 = get<0, 2>(X);
        auto& x03 = get<0, 3>(X);
        auto& x10 = get<1, 0>(X);
        auto& x11 = get<1, 1>(X);
        auto& x12 = get<1, 2>(X);
        auto& x13 = get<1, 3>(X);
        auto& x20 = get<2, 0>(X);
        auto& x21 = get<2, 1>(X);
        auto& x22 = get<2, 2>(X);
        auto& x23 = get<2, 3>(X);
        auto& x30 = get<3, 0>(X);
        auto& x31 = get<3, 1>(X);
        auto& x32 = get<3, 2>(X);
        auto& x33 = get<3, 3>(X);
        // Phase 1: streaming minors → X_0 in buffer (inverse_det-in-x33: x33 =
        // 1/det throughout; set LAST in Group 3 via x33 *= cofactor(3,3))
        {  // Group 1: m23 (rows 2,3) → det, inverse_det(x33), 8 X_0 entries
          auto& m23_23 = get<0, 0>(R);
          auto& m23_13 = get<0, 1>(R);
          auto& m23_03 = get<0, 2>(R);
          auto& m23_12 = get<0, 3>(R);
          auto& m23_02 = get<1, 0>(R);
          auto& m23_01 = get<1, 1>(R);
          m23_23 = t22 * t33 - t23 * t32;
          m23_13 = t21 * t33 - t23 * t31;
          m23_03 = t20 * t33 - t23 * t30;
          m23_12 = t21 * t32 - t22 * t31;
          m23_02 = t20 * t32 - t22 * t30;
          m23_01 = t20 * t31 - t21 * t30;
          get(*det) = t00 * (t11 * m23_23 - t12 * m23_13 + t13 * m23_12) -
                      t01 * (t10 * m23_23 - t12 * m23_03 + t13 * m23_02) +
                      t02 * (t10 * m23_13 - t11 * m23_03 + t13 * m23_01) -
                      t03 * (t10 * m23_12 - t11 * m23_02 + t12 * m23_01);
          x33 = 1.0 / get(*det);  // inverse_det; set LAST in Group 3
          x00 = (t11 * m23_23 - t12 * m23_13 + t13 * m23_12) * x33;
          x10 = -(t10 * m23_23 - t12 * m23_03 + t13 * m23_02) * x33;
          x20 = (t10 * m23_13 - t11 * m23_03 + t13 * m23_01) * x33;
          x30 = -(t10 * m23_12 - t11 * m23_02 + t12 * m23_01) * x33;
          x01 = -(t01 * m23_23 - t02 * m23_13 + t03 * m23_12) * x33;
          x11 = (t00 * m23_23 - t02 * m23_03 + t03 * m23_02) * x33;
          x21 = -(t00 * m23_13 - t01 * m23_03 + t03 * m23_01) * x33;
          x31 = (t00 * m23_12 - t01 * m23_02 + t02 * m23_01) * x33;
        }
        {  // Group 2: reuse R[0,0..1,1] for m13 (rows 1,3) → 4 X_0 entries
          auto& m13_23 = get<0, 0>(R);
          auto& m13_13 = get<0, 1>(R);
          auto& m13_03 = get<0, 2>(R);
          auto& m13_12 = get<0, 3>(R);
          auto& m13_02 = get<1, 0>(R);
          auto& m13_01 = get<1, 1>(R);
          m13_23 = t12 * t33 - t13 * t32;
          m13_13 = t11 * t33 - t13 * t31;
          m13_03 = t10 * t33 - t13 * t30;
          m13_12 = t11 * t32 - t12 * t31;
          m13_02 = t10 * t32 - t12 * t30;
          m13_01 = t10 * t31 - t11 * t30;
          x02 = (t01 * m13_23 - t02 * m13_13 + t03 * m13_12) * x33;
          x12 = -(t00 * m13_23 - t02 * m13_03 + t03 * m13_02) * x33;
          x22 = (t00 * m13_13 - t01 * m13_03 + t03 * m13_01) * x33;
          x32 = -(t00 * m13_12 - t01 * m13_02 + t02 * m13_01) * x33;
        }
        {  // Group 3: reuse R[0,0..1,1] for m12 (rows 1,2) → x03,x13,x23;
           // x33 *= cof(3,3) — set LAST
          auto& m12_23 = get<0, 0>(R);
          auto& m12_13 = get<0, 1>(R);
          auto& m12_03 = get<0, 2>(R);
          auto& m12_12 = get<0, 3>(R);
          auto& m12_02 = get<1, 0>(R);
          auto& m12_01 = get<1, 1>(R);
          m12_23 = t12 * t23 - t13 * t22;
          m12_13 = t11 * t23 - t13 * t21;
          m12_03 = t10 * t23 - t13 * t20;
          m12_12 = t11 * t22 - t12 * t21;
          m12_02 = t10 * t22 - t12 * t20;
          m12_01 = t10 * t21 - t11 * t20;
          x03 = -(t01 * m12_23 - t02 * m12_13 + t03 * m12_12) * x33;
          x13 = (t00 * m12_23 - t02 * m12_03 + t03 * m12_02) * x33;
          x23 = -(t00 * m12_13 - t01 * m12_03 + t03 * m12_01) * x33;
          x33 *= (t00 * m12_12 - t01 * m12_02 + t02 * m12_01);  // set LAST
        }
        // Phase 2: R = I - A*X_0 fused into R tensor (reads X_0 from buffer)
        auto& r00 = get<0, 0>(R);
        auto& r01 = get<0, 1>(R);
        auto& r02 = get<0, 2>(R);
        auto& r03 = get<0, 3>(R);
        auto& r10 = get<1, 0>(R);
        auto& r11 = get<1, 1>(R);
        auto& r12 = get<1, 2>(R);
        auto& r13 = get<1, 3>(R);
        auto& r20 = get<2, 0>(R);
        auto& r21 = get<2, 1>(R);
        auto& r22 = get<2, 2>(R);
        auto& r23 = get<2, 3>(R);
        auto& r30 = get<3, 0>(R);
        auto& r31 = get<3, 1>(R);
        auto& r32 = get<3, 2>(R);
        auto& r33 = get<3, 3>(R);
        r00 = 1.0 - (t00 * x00 + t01 * x10 + t02 * x20 + t03 * x30);
        r01 = -(t00 * x01 + t01 * x11 + t02 * x21 + t03 * x31);
        r02 = -(t00 * x02 + t01 * x12 + t02 * x22 + t03 * x32);
        r03 = -(t00 * x03 + t01 * x13 + t02 * x23 + t03 * x33);
        r10 = -(t10 * x00 + t11 * x10 + t12 * x20 + t13 * x30);
        r11 = 1.0 - (t10 * x01 + t11 * x11 + t12 * x21 + t13 * x31);
        r12 = -(t10 * x02 + t11 * x12 + t12 * x22 + t13 * x32);
        r13 = -(t10 * x03 + t11 * x13 + t12 * x23 + t13 * x33);
        r20 = -(t20 * x00 + t21 * x10 + t22 * x20 + t23 * x30);
        r21 = -(t20 * x01 + t21 * x11 + t22 * x21 + t23 * x31);
        r22 = 1.0 - (t20 * x02 + t21 * x12 + t22 * x22 + t23 * x32);
        r23 = -(t20 * x03 + t21 * x13 + t22 * x23 + t23 * x33);
        r30 = -(t30 * x00 + t31 * x10 + t32 * x20 + t33 * x30);
        r31 = -(t30 * x01 + t31 * x11 + t32 * x21 + t33 * x31);
        r32 = -(t30 * x02 + t31 * x12 + t32 * x22 + t33 * x32);
        r33 = 1.0 - (t30 * x03 + t31 * x13 + t32 * x23 + t33 * x33);
        // Phase 3: Y = X_0 + X_0*R → *inv (no aliasing with buffer)
        get<0, 0>(*inv) = x00 + x00 * r00 + x01 * r10 + x02 * r20 + x03 * r30;
        get<0, 1>(*inv) = x01 + x00 * r01 + x01 * r11 + x02 * r21 + x03 * r31;
        get<0, 2>(*inv) = x02 + x00 * r02 + x01 * r12 + x02 * r22 + x03 * r32;
        get<0, 3>(*inv) = x03 + x00 * r03 + x01 * r13 + x02 * r23 + x03 * r33;
        get<1, 0>(*inv) = x10 + x10 * r00 + x11 * r10 + x12 * r20 + x13 * r30;
        get<1, 1>(*inv) = x11 + x10 * r01 + x11 * r11 + x12 * r21 + x13 * r31;
        get<1, 2>(*inv) = x12 + x10 * r02 + x11 * r12 + x12 * r22 + x13 * r32;
        get<1, 3>(*inv) = x13 + x10 * r03 + x11 * r13 + x12 * r23 + x13 * r33;
        get<2, 0>(*inv) = x20 + x20 * r00 + x21 * r10 + x22 * r20 + x23 * r30;
        get<2, 1>(*inv) = x21 + x20 * r01 + x21 * r11 + x22 * r21 + x23 * r31;
        get<2, 2>(*inv) = x22 + x20 * r02 + x21 * r12 + x22 * r22 + x23 * r32;
        get<2, 3>(*inv) = x23 + x20 * r03 + x21 * r13 + x22 * r23 + x23 * r33;
        get<3, 0>(*inv) = x30 + x30 * r00 + x31 * r10 + x32 * r20 + x33 * r30;
        get<3, 1>(*inv) = x31 + x30 * r01 + x31 * r11 + x32 * r21 + x33 * r31;
        get<3, 2>(*inv) = x32 + x30 * r02 + x31 * r12 + x32 * r22 + x33 * r32;
        get<3, 3>(*inv) = x33 + x30 * r03 + x31 * r13 + x32 * r23 + x33 * r33;
        // Phase 4: det = 1/det(Y) via 6 minors. Reuse X entries as scratch.
        const T& y00 = get<0, 0>(*inv);
        const T& y01 = get<0, 1>(*inv);
        const T& y02 = get<0, 2>(*inv);
        const T& y03 = get<0, 3>(*inv);
        const T& y10 = get<1, 0>(*inv);
        const T& y11 = get<1, 1>(*inv);
        const T& y12 = get<1, 2>(*inv);
        const T& y13 = get<1, 3>(*inv);
        const T& y20 = get<2, 0>(*inv);
        const T& y21 = get<2, 1>(*inv);
        const T& y22 = get<2, 2>(*inv);
        const T& y23 = get<2, 3>(*inv);
        const T& y30 = get<3, 0>(*inv);
        const T& y31 = get<3, 1>(*inv);
        const T& y32 = get<3, 2>(*inv);
        const T& y33 = get<3, 3>(*inv);
        auto& n23_23 = get<0, 0>(X);
        auto& n23_13 = get<0, 1>(X);
        auto& n23_03 = get<0, 2>(X);
        auto& n23_12 = get<0, 3>(X);
        auto& n23_02 = get<1, 0>(X);
        auto& n23_01 = get<1, 1>(X);
        auto& det_y = get<1, 2>(X);
        n23_23 = y22 * y33 - y23 * y32;
        n23_13 = y21 * y33 - y23 * y31;
        n23_03 = y20 * y33 - y23 * y30;
        n23_12 = y21 * y32 - y22 * y31;
        n23_02 = y20 * y32 - y22 * y30;
        n23_01 = y20 * y31 - y21 * y30;
        det_y = y00 * (y11 * n23_23 - y12 * n23_13 + y13 * n23_12) -
                y01 * (y10 * n23_23 - y12 * n23_03 + y13 * n23_02) +
                y02 * (y10 * n23_13 - y11 * n23_03 + y13 * n23_01) -
                y03 * (y10 * n23_12 - y11 * n23_02 + y12 * n23_01);
        get(*det) = 1.0 / det_y;
      } else {
        // Analytic: 6 slots, streaming minors.
        // inv[3,3] holds inverse_det = 1/det; multiplied by its cofactor last.
        // Group 1 (m23) → det + inv cols 0-1; slots reused for Group 2 (m13)
        // → inv col 2; slots reused for Group 3 (m12) → inv col 3.
        TempBuffer<tmpl::list<Tags::TempScalar<0, T>, Tags::TempScalar<1, T>,
                              Tags::TempScalar<2, T>, Tags::TempScalar<3, T>,
                              Tags::TempScalar<4, T>, Tags::TempScalar<5, T>>>
            buffer(get_size(get<0, 0>(tensor)));
        // Group 1: m23 minors → det + inv cols 0-1
        auto& m23_23 = get(get<Tags::TempScalar<0, T>>(buffer));
        auto& m23_13 = get(get<Tags::TempScalar<1, T>>(buffer));
        auto& m23_03 = get(get<Tags::TempScalar<2, T>>(buffer));
        auto& m23_12 = get(get<Tags::TempScalar<3, T>>(buffer));
        auto& m23_02 = get(get<Tags::TempScalar<4, T>>(buffer));
        auto& m23_01 = get(get<Tags::TempScalar<5, T>>(buffer));
        m23_23 = t22 * t33 - t23 * t32;
        m23_13 = t21 * t33 - t23 * t31;
        m23_03 = t20 * t33 - t23 * t30;
        m23_12 = t21 * t32 - t22 * t31;
        m23_02 = t20 * t32 - t22 * t30;
        m23_01 = t20 * t31 - t21 * t30;
        get(*det) = t00 * (t11 * m23_23 - t12 * m23_13 + t13 * m23_12) -
                    t01 * (t10 * m23_23 - t12 * m23_03 + t13 * m23_02) +
                    t02 * (t10 * m23_13 - t11 * m23_03 + t13 * m23_01) -
                    t03 * (t10 * m23_12 - t11 * m23_02 + t12 * m23_01);
        get<3, 3>(*inv) = 1.0 / get(*det);  // inverse_det; set last
        get<0, 0>(*inv) =
            (t11 * m23_23 - t12 * m23_13 + t13 * m23_12) * get<3, 3>(*inv);
        get<1, 0>(*inv) =
            -(t10 * m23_23 - t12 * m23_03 + t13 * m23_02) * get<3, 3>(*inv);
        get<2, 0>(*inv) =
            (t10 * m23_13 - t11 * m23_03 + t13 * m23_01) * get<3, 3>(*inv);
        get<3, 0>(*inv) =
            -(t10 * m23_12 - t11 * m23_02 + t12 * m23_01) * get<3, 3>(*inv);
        get<0, 1>(*inv) =
            -(t01 * m23_23 - t02 * m23_13 + t03 * m23_12) * get<3, 3>(*inv);
        get<1, 1>(*inv) =
            (t00 * m23_23 - t02 * m23_03 + t03 * m23_02) * get<3, 3>(*inv);
        get<2, 1>(*inv) =
            -(t00 * m23_13 - t01 * m23_03 + t03 * m23_01) * get<3, 3>(*inv);
        get<3, 1>(*inv) =
            (t00 * m23_12 - t01 * m23_02 + t02 * m23_01) * get<3, 3>(*inv);
        // Group 2: reuse same 6 slots for m13 minors → inv col 2
        auto& m13_23 = get(get<Tags::TempScalar<0, T>>(buffer));
        auto& m13_13 = get(get<Tags::TempScalar<1, T>>(buffer));
        auto& m13_03 = get(get<Tags::TempScalar<2, T>>(buffer));
        auto& m13_12 = get(get<Tags::TempScalar<3, T>>(buffer));
        auto& m13_02 = get(get<Tags::TempScalar<4, T>>(buffer));
        auto& m13_01 = get(get<Tags::TempScalar<5, T>>(buffer));
        m13_23 = t12 * t33 - t13 * t32;
        m13_13 = t11 * t33 - t13 * t31;
        m13_03 = t10 * t33 - t13 * t30;
        m13_12 = t11 * t32 - t12 * t31;
        m13_02 = t10 * t32 - t12 * t30;
        m13_01 = t10 * t31 - t11 * t30;
        get<0, 2>(*inv) =
            (t01 * m13_23 - t02 * m13_13 + t03 * m13_12) * get<3, 3>(*inv);
        get<1, 2>(*inv) =
            -(t00 * m13_23 - t02 * m13_03 + t03 * m13_02) * get<3, 3>(*inv);
        get<2, 2>(*inv) =
            (t00 * m13_13 - t01 * m13_03 + t03 * m13_01) * get<3, 3>(*inv);
        get<3, 2>(*inv) =
            -(t00 * m13_12 - t01 * m13_02 + t02 * m13_01) * get<3, 3>(*inv);
        // Group 3: reuse same 6 slots for m12 minors → inv col 3
        auto& m12_23 = get(get<Tags::TempScalar<0, T>>(buffer));
        auto& m12_13 = get(get<Tags::TempScalar<1, T>>(buffer));
        auto& m12_03 = get(get<Tags::TempScalar<2, T>>(buffer));
        auto& m12_12 = get(get<Tags::TempScalar<3, T>>(buffer));
        auto& m12_02 = get(get<Tags::TempScalar<4, T>>(buffer));
        auto& m12_01 = get(get<Tags::TempScalar<5, T>>(buffer));
        m12_23 = t12 * t23 - t13 * t22;
        m12_13 = t11 * t23 - t13 * t21;
        m12_03 = t10 * t23 - t13 * t20;
        m12_12 = t11 * t22 - t12 * t21;
        m12_02 = t10 * t22 - t12 * t20;
        m12_01 = t10 * t21 - t11 * t20;
        get<0, 3>(*inv) =
            -(t01 * m12_23 - t02 * m12_13 + t03 * m12_12) * get<3, 3>(*inv);
        get<1, 3>(*inv) =
            (t00 * m12_23 - t02 * m12_03 + t03 * m12_02) * get<3, 3>(*inv);
        get<2, 3>(*inv) =
            -(t00 * m12_13 - t01 * m12_03 + t03 * m12_01) * get<3, 3>(*inv);
        get<3, 3>(*inv) *=
            (t00 * m12_12 - t01 * m12_02 + t02 * m12_01);  // set LAST
      }
    }
  }
};

// 4x4 symmetric (Symmetry<1,1>):
// Same three-way dispatch as the generic case. The symmetric specialization
// exploits the symmetry of both the input tensor and its inverse.
template <typename Index0>
struct DetAndInverseImpl<Symmetry<1, 1>, Index0, Index0,
                         Requires<Index0::dim == 4>> {
  template <InversionMethod Method, typename T>
  static void apply(
      const gsl::not_null<Scalar<T>*> det,
      const gsl::not_null<
          Tensor<T, Symmetry<1, 1>, inverse_indices<Index0, Index0>>*>
          inv,
      const Tensor<T, Symmetry<1, 1>, tmpl::list<Index0, Index0>>& tensor) {
    if constexpr (Method == InversionMethod::BlockTopLeft) {
      const T& p00 = get<0, 0>(tensor);
      const T& p01 = get<0, 1>(tensor);
      const T& p11 = get<1, 1>(tensor);
      const T& q00 = get<0, 2>(tensor);
      const T& q01 = get<0, 3>(tensor);
      const T& q10 = get<1, 2>(tensor);
      const T& q11 = get<1, 3>(tensor);
      const T& s00 = get<2, 2>(tensor);
      const T& s01 = get<2, 3>(tensor);
      const T& s11 = get<3, 3>(tensor);

      T& u00 = get<0, 0>(*inv);
      T& u01 = get<0, 1>(*inv);
      T& u11 = get<1, 1>(*inv);
      T& v00 = get<0, 2>(*inv);
      T& v01 = get<0, 3>(*inv);
      T& v10 = get<1, 2>(*inv);
      T& v11 = get<1, 3>(*inv);
      T& x00 = get<2, 2>(*inv);
      T& x01 = get<2, 3>(*inv);
      T& x11 = get<3, 3>(*inv);

      // Temporarily store det(P) in det
      get(*det) = p00 * p11 - p01 * p01;
      // 7 slots: 0-2=inv_p; 3-6=r_inv_p.
      // inverse_det_p in u00 (overwritten later); Schur complement in x block;
      // x inversion uses u00, u01 as scratch (both overwritten in final step).
      // 7 slots: inv_p (3 sym) in Tempij<0,2>; r_inv_p (4) in TempIj<0,2>.
      // inverse_det_p in u00 (overwritten later); Schur complement in x block;
      // x inversion uses u00, u01 as scratch (both overwritten in final step).
      TempBuffer<tmpl::list<Tags::Tempij<0, 2, Frame::Inertial, T>,
                            Tags::TempIj<0, 2, Frame::Inertial, T>>>
          buffer(get_size(get<0, 0>(tensor)));
      u00 = 1.0 / get(*det);  // inverse_det_p; overwritten later
      auto& inv_p = get<Tags::Tempij<0, 2, Frame::Inertial, T>>(buffer);
      auto& inv_p00 = get<0, 0>(inv_p);
      auto& inv_p01 = get<0, 1>(inv_p);
      auto& inv_p11 = get<1, 1>(inv_p);
      inv_p00 = p11 * u00;
      inv_p01 = -p01 * u00;
      inv_p11 = p00 * u00;
      // R = Q^T for symmetric tensors
      auto& r_inv_p = get<Tags::TempIj<0, 2, Frame::Inertial, T>>(buffer);
      auto& r_inv_p00 = get<0, 0>(r_inv_p);
      auto& r_inv_p01 = get<0, 1>(r_inv_p);
      auto& r_inv_p10 = get<1, 0>(r_inv_p);
      auto& r_inv_p11 = get<1, 1>(r_inv_p);
      r_inv_p00 = q00 * inv_p00 + q10 * inv_p01;
      r_inv_p01 = q00 * inv_p01 + q10 * inv_p11;
      r_inv_p10 = q01 * inv_p00 + q11 * inv_p01;
      r_inv_p11 = q01 * inv_p01 + q11 * inv_p11;
      // Schur complement → x block directly
      x00 = s00 - (r_inv_p00 * q00 + r_inv_p01 * q10);
      x01 = s01 - (r_inv_p00 * q01 + r_inv_p01 * q11);
      x11 = s11 - (r_inv_p10 * q01 + r_inv_p11 * q11);
      // Invert x in-place: save x00 in u00, det(x) in u01 (both overwritten
      // in the final u-block step below)
      u00 = x00;
      u01 = x00 * x11 - x01 * x01;
      get(*det) *= u01;
      u01 = 1.0 / u01;
      x00 = x11 * u01;
      x01 = -x01 * u01;
      x11 = u00 * u01;
      v00 = -x00 * r_inv_p00 - x01 * r_inv_p10;
      v10 = -x00 * r_inv_p01 - x01 * r_inv_p11;
      v01 = -x01 * r_inv_p00 - x11 * r_inv_p10;
      v11 = -x01 * r_inv_p01 - x11 * r_inv_p11;
      u00 = inv_p00 - (r_inv_p00 * v00 + r_inv_p10 * v01);
      u01 = inv_p01 - (r_inv_p00 * v10 + r_inv_p10 * v11);
      u11 = inv_p11 - (r_inv_p01 * v10 + r_inv_p11 * v11);

    } else if constexpr (Method == InversionMethod::BlockBottomRight) {
      // For symmetric tensors R = Q^T, so inv(S)*R = (Q*inv(S))^T = q_inv_s^T
      const T& p00 = get<0, 0>(tensor);
      const T& p01 = get<0, 1>(tensor);
      const T& p11 = get<1, 1>(tensor);
      const T& q00 = get<0, 2>(tensor);
      const T& q01 = get<0, 3>(tensor);
      const T& q10 = get<1, 2>(tensor);
      const T& q11 = get<1, 3>(tensor);
      const T& s00 = get<2, 2>(tensor);
      const T& s01 = get<2, 3>(tensor);
      const T& s11 = get<3, 3>(tensor);

      T& u00 = get<0, 0>(*inv);
      T& u01 = get<0, 1>(*inv);
      T& u11 = get<1, 1>(*inv);
      T& v00 = get<0, 2>(*inv);
      T& v01 = get<0, 3>(*inv);
      T& v10 = get<1, 2>(*inv);
      T& v11 = get<1, 3>(*inv);
      T& x00 = get<2, 2>(*inv);
      T& x01 = get<2, 3>(*inv);
      T& x11 = get<3, 3>(*inv);

      // Temporarily store det(S) in det
      get(*det) = s00 * s11 - s01 * s01;
      // 7 slots: inv_s (3 sym) in Tempij<0,2>; q_inv_s (4) in TempIj<0,2>.
      // inverse_det_s in x00 (overwritten later); Schur complement in u block;
      // u inversion uses x00, x01 as scratch (both overwritten in final step).
      TempBuffer<tmpl::list<Tags::Tempij<0, 2, Frame::Inertial, T>,
                            Tags::TempIj<0, 2, Frame::Inertial, T>>>
          buffer(get_size(get<0, 0>(tensor)));
      x00 = 1.0 / get(*det);  // inverse_det_s; overwritten later
      auto& inv_s = get<Tags::Tempij<0, 2, Frame::Inertial, T>>(buffer);
      auto& inv_s00 = get<0, 0>(inv_s);
      auto& inv_s01 = get<0, 1>(inv_s);
      auto& inv_s11 = get<1, 1>(inv_s);
      inv_s00 = s11 * x00;
      inv_s01 = -s01 * x00;
      inv_s11 = s00 * x00;
      // Q * inv(S)  (symmetric inv(S): inv_s10 = inv_s01)
      auto& q_inv_s = get<Tags::TempIj<0, 2, Frame::Inertial, T>>(buffer);
      auto& q_inv_s00 = get<0, 0>(q_inv_s);
      auto& q_inv_s01 = get<0, 1>(q_inv_s);
      auto& q_inv_s10 = get<1, 0>(q_inv_s);
      auto& q_inv_s11 = get<1, 1>(q_inv_s);
      q_inv_s00 = q00 * inv_s00 + q01 * inv_s01;
      q_inv_s01 = q00 * inv_s01 + q01 * inv_s11;
      q_inv_s10 = q10 * inv_s00 + q11 * inv_s01;
      q_inv_s11 = q10 * inv_s01 + q11 * inv_s11;
      // Schur complement of S → u block directly (R = Q^T for symmetric)
      u00 = p00 - (q_inv_s00 * q00 + q_inv_s01 * q01);
      u01 = p01 - (q_inv_s00 * q10 + q_inv_s01 * q11);
      u11 = p11 - (q_inv_s10 * q10 + q_inv_s11 * q11);
      // Invert u in-place: save u00 in x00, det(u) in x01 (both overwritten
      // in the final x-block step below)
      x00 = u00;
      x01 = u00 * u11 - u01 * u01;
      get(*det) *= x01;
      x01 = 1.0 / x01;
      u00 = u11 * x01;
      u01 = -u01 * x01;
      u11 = x00 * x01;
      // V = -U * Q * inv(S)
      v00 = -(u00 * q_inv_s00 + u01 * q_inv_s10);
      v01 = -(u00 * q_inv_s01 + u01 * q_inv_s11);
      v10 = -(u01 * q_inv_s00 + u11 * q_inv_s10);
      v11 = -(u01 * q_inv_s01 + u11 * q_inv_s11);
      // X = inv(S) - V^T * q_inv_s  (W = V^T for symmetric case)
      x00 = inv_s00 - (v00 * q_inv_s00 + v10 * q_inv_s10);
      x01 = inv_s01 - (v00 * q_inv_s01 + v10 * q_inv_s11);
      x11 = inv_s11 - (v01 * q_inv_s01 + v11 * q_inv_s11);

    } else {
      // Analytic (and Refined, which adds a Newton-Schulz step after):
      // direct Cramer's rule. For symmetric tensors, t_{ji} = t_{ij} and
      // the inverse is also symmetric, so only 10 independent components
      // are stored.
      const T& t00 = get<0, 0>(tensor);
      const T& t01 = get<0, 1>(tensor);
      const T& t02 = get<0, 2>(tensor);
      const T& t03 = get<0, 3>(tensor);
      const T& t11 = get<1, 1>(tensor);
      const T& t12 = get<1, 2>(tensor);
      const T& t13 = get<1, 3>(tensor);
      const T& t22 = get<2, 2>(tensor);
      const T& t23 = get<2, 3>(tensor);
      const T& t33 = get<3, 3>(tensor);
      if constexpr (Method == InversionMethod::Refined) {
        // 26 slots: Phase 1 streams minors (slots 10-15, reused per group),
        //   inverse_det (slot 16), writing X_0 directly to slots 0-9;
        //   Phase 2: X_0 in 0-9, R = I-AX in 10-25;
        //   off-diagonal C pair inlined; Phase 4 reuses 0-6 for minors+det.
        TempBuffer<tmpl::list<Tags::TempScalar<0, T>, Tags::TempScalar<1, T>,
                              Tags::TempScalar<2, T>, Tags::TempScalar<3, T>,
                              Tags::TempScalar<4, T>, Tags::TempScalar<5, T>,
                              Tags::TempScalar<6, T>, Tags::TempScalar<7, T>,
                              Tags::TempScalar<8, T>, Tags::TempScalar<9, T>,
                              Tags::TempScalar<10, T>, Tags::TempScalar<11, T>,
                              Tags::TempScalar<12, T>, Tags::TempScalar<13, T>,
                              Tags::TempScalar<14, T>, Tags::TempScalar<15, T>,
                              Tags::TempScalar<16, T>, Tags::TempScalar<17, T>,
                              Tags::TempScalar<18, T>, Tags::TempScalar<19, T>,
                              Tags::TempScalar<20, T>, Tags::TempScalar<21, T>,
                              Tags::TempScalar<22, T>, Tags::TempScalar<23, T>,
                              Tags::TempScalar<24, T>, Tags::TempScalar<25, T>>>
            buffer(get_size(get<0, 0>(tensor)));
        // Phase 1: Streaming minors → X_0 directly to buffer slots 0-9.
        // Minors reuse slots 10-15 per group; inverse_det in slot 16.
        // Each cofactor uses minors from only one row-pair group.
        auto& x00 = get(get<Tags::TempScalar<0, T>>(buffer));
        auto& x01 = get(get<Tags::TempScalar<1, T>>(buffer));
        auto& x02 = get(get<Tags::TempScalar<2, T>>(buffer));
        auto& x03 = get(get<Tags::TempScalar<3, T>>(buffer));
        auto& x11 = get(get<Tags::TempScalar<4, T>>(buffer));
        auto& x12 = get(get<Tags::TempScalar<5, T>>(buffer));
        auto& x13 = get(get<Tags::TempScalar<6, T>>(buffer));
        auto& x22 = get(get<Tags::TempScalar<7, T>>(buffer));
        auto& x23 = get(get<Tags::TempScalar<8, T>>(buffer));
        auto& x33 = get(get<Tags::TempScalar<9, T>>(buffer));
        auto& inverse_det = get(get<Tags::TempScalar<16, T>>(buffer));
        {  // Group 1: m23 → det, inverse_det, x00(0), x01(1), x11(4)
          auto& m23_23 = get(get<Tags::TempScalar<10, T>>(buffer));
          auto& m23_13 = get(get<Tags::TempScalar<11, T>>(buffer));
          auto& m23_03 = get(get<Tags::TempScalar<12, T>>(buffer));
          auto& m23_12 = get(get<Tags::TempScalar<13, T>>(buffer));
          auto& m23_02 = get(get<Tags::TempScalar<14, T>>(buffer));
          auto& m23_01 = get(get<Tags::TempScalar<15, T>>(buffer));
          m23_23 = t22 * t33 - t23 * t23;
          m23_13 = t12 * t33 - t23 * t13;
          m23_03 = t02 * t33 - t23 * t03;
          m23_12 = t12 * t23 - t22 * t13;
          m23_02 = t02 * t23 - t22 * t03;
          m23_01 = t02 * t13 - t12 * t03;
          get(*det) = t00 * (t11 * m23_23 - t12 * m23_13 + t13 * m23_12) -
                      t01 * (t01 * m23_23 - t12 * m23_03 + t13 * m23_02) +
                      t02 * (t01 * m23_13 - t11 * m23_03 + t13 * m23_01) -
                      t03 * (t01 * m23_12 - t11 * m23_02 + t12 * m23_01);
          inverse_det = 1.0 / get(*det);
          x00 = (t11 * m23_23 - t12 * m23_13 + t13 * m23_12) * inverse_det;
          x01 = -(t01 * m23_23 - t02 * m23_13 + t03 * m23_12) * inverse_det;
          x11 = (t00 * m23_23 - t02 * m23_03 + t03 * m23_02) * inverse_det;
        }
        {  // Group 2: m13 → x02(2), x12(5), x22(7)
          auto& m13_23 = get(get<Tags::TempScalar<10, T>>(buffer));
          auto& m13_13 = get(get<Tags::TempScalar<11, T>>(buffer));
          auto& m13_03 = get(get<Tags::TempScalar<12, T>>(buffer));
          auto& m13_12 = get(get<Tags::TempScalar<13, T>>(buffer));
          auto& m13_02 = get(get<Tags::TempScalar<14, T>>(buffer));
          auto& m13_01 = get(get<Tags::TempScalar<15, T>>(buffer));
          m13_23 = t12 * t33 - t13 * t23;
          m13_13 = t11 * t33 - t13 * t13;
          m13_03 = t01 * t33 - t13 * t03;
          m13_12 = t11 * t23 - t12 * t13;
          m13_02 = t01 * t23 - t12 * t03;
          m13_01 = t01 * t13 - t11 * t03;
          x02 = (t01 * m13_23 - t02 * m13_13 + t03 * m13_12) * inverse_det;
          x12 = -(t00 * m13_23 - t02 * m13_03 + t03 * m13_02) * inverse_det;
          x22 = (t00 * m13_13 - t01 * m13_03 + t03 * m13_01) * inverse_det;
        }
        {  // Group 3: m12 → x03(3), x13(6), x23(8), x33(9)
          auto& m12_23 = get(get<Tags::TempScalar<10, T>>(buffer));
          auto& m12_13 = get(get<Tags::TempScalar<11, T>>(buffer));
          auto& m12_03 = get(get<Tags::TempScalar<12, T>>(buffer));
          auto& m12_12 = get(get<Tags::TempScalar<13, T>>(buffer));
          auto& m12_02 = get(get<Tags::TempScalar<14, T>>(buffer));
          auto& m12_01 = get(get<Tags::TempScalar<15, T>>(buffer));
          m12_23 = t12 * t23 - t13 * t22;
          m12_13 = t11 * t23 - t13 * t12;
          m12_03 = t01 * t23 - t13 * t02;
          m12_12 = t11 * t22 - t12 * t12;
          m12_02 = t01 * t22 - t12 * t02;
          m12_01 = t01 * t12 - t11 * t02;
          x03 = -(t01 * m12_23 - t02 * m12_13 + t03 * m12_12) * inverse_det;
          x13 = (t00 * m12_23 - t02 * m12_03 + t03 * m12_02) * inverse_det;
          x23 = -(t00 * m12_13 - t01 * m12_03 + t03 * m12_01) * inverse_det;
          x33 = (t00 * m12_12 - t01 * m12_02 + t02 * m12_01) * inverse_det;
        }
        // Phase 2: R = I - A*X_0 fused into slots 10-25 (A sym, X_0 sym)
        auto& ax00 = get(get<Tags::TempScalar<10, T>>(buffer));
        auto& ax01 = get(get<Tags::TempScalar<11, T>>(buffer));
        auto& ax02 = get(get<Tags::TempScalar<12, T>>(buffer));
        auto& ax03 = get(get<Tags::TempScalar<13, T>>(buffer));
        auto& ax10 = get(get<Tags::TempScalar<14, T>>(buffer));
        auto& ax11 = get(get<Tags::TempScalar<15, T>>(buffer));
        auto& ax12 = get(get<Tags::TempScalar<16, T>>(buffer));
        auto& ax13 = get(get<Tags::TempScalar<17, T>>(buffer));
        auto& ax20 = get(get<Tags::TempScalar<18, T>>(buffer));
        auto& ax21 = get(get<Tags::TempScalar<19, T>>(buffer));
        auto& ax22 = get(get<Tags::TempScalar<20, T>>(buffer));
        auto& ax23 = get(get<Tags::TempScalar<21, T>>(buffer));
        auto& ax30 = get(get<Tags::TempScalar<22, T>>(buffer));
        auto& ax31 = get(get<Tags::TempScalar<23, T>>(buffer));
        auto& ax32 = get(get<Tags::TempScalar<24, T>>(buffer));
        auto& ax33 = get(get<Tags::TempScalar<25, T>>(buffer));
        ax00 = 1.0 - (t00 * x00 + t01 * x01 + t02 * x02 + t03 * x03);
        ax01 = -(t00 * x01 + t01 * x11 + t02 * x12 + t03 * x13);
        ax02 = -(t00 * x02 + t01 * x12 + t02 * x22 + t03 * x23);
        ax03 = -(t00 * x03 + t01 * x13 + t02 * x23 + t03 * x33);
        ax10 = -(t01 * x00 + t11 * x01 + t12 * x02 + t13 * x03);
        ax11 = 1.0 - (t01 * x01 + t11 * x11 + t12 * x12 + t13 * x13);
        ax12 = -(t01 * x02 + t11 * x12 + t12 * x22 + t13 * x23);
        ax13 = -(t01 * x03 + t11 * x13 + t12 * x23 + t13 * x33);
        ax20 = -(t02 * x00 + t12 * x01 + t22 * x02 + t23 * x03);
        ax21 = -(t02 * x01 + t12 * x11 + t22 * x12 + t23 * x13);
        ax22 = 1.0 - (t02 * x02 + t12 * x12 + t22 * x22 + t23 * x23);
        ax23 = -(t02 * x03 + t12 * x13 + t22 * x23 + t23 * x33);
        ax30 = -(t03 * x00 + t13 * x01 + t23 * x02 + t33 * x03);
        ax31 = -(t03 * x01 + t13 * x11 + t23 * x12 + t33 * x13);
        ax32 = -(t03 * x02 + t13 * x12 + t23 * x22 + t33 * x23);
        ax33 = 1.0 - (t03 * x03 + t13 * x13 + t23 * x23 + t33 * x33);
        // Diagonal: Y_ii = x_ii + C_ii (no pair needed)
        get<0, 0>(*inv) =
            x00 + (x00 * ax00 + x01 * ax10 + x02 * ax20 + x03 * ax30);
        get<1, 1>(*inv) =
            x11 + (x01 * ax01 + x11 * ax11 + x12 * ax21 + x13 * ax31);
        get<2, 2>(*inv) =
            x22 + (x02 * ax02 + x12 * ax12 + x22 * ax22 + x23 * ax32);
        get<3, 3>(*inv) =
            x33 + (x03 * ax03 + x13 * ax13 + x23 * ax23 + x33 * ax33);
        // Off-diagonal: inline C_ij + C_ji for symmetrization
        get<0, 1>(*inv) =
            x01 + 0.5 * (x00 * ax01 + x01 * ax11 + x02 * ax21 + x03 * ax31 +
                         x01 * ax00 + x11 * ax10 + x12 * ax20 + x13 * ax30);
        get<0, 2>(*inv) =
            x02 + 0.5 * (x00 * ax02 + x01 * ax12 + x02 * ax22 + x03 * ax32 +
                         x02 * ax00 + x12 * ax10 + x22 * ax20 + x23 * ax30);
        get<0, 3>(*inv) =
            x03 + 0.5 * (x00 * ax03 + x01 * ax13 + x02 * ax23 + x03 * ax33 +
                         x03 * ax00 + x13 * ax10 + x23 * ax20 + x33 * ax30);
        get<1, 2>(*inv) =
            x12 + 0.5 * (x01 * ax02 + x11 * ax12 + x12 * ax22 + x13 * ax32 +
                         x02 * ax01 + x12 * ax11 + x22 * ax21 + x23 * ax31);
        get<1, 3>(*inv) =
            x13 + 0.5 * (x01 * ax03 + x11 * ax13 + x12 * ax23 + x13 * ax33 +
                         x03 * ax01 + x13 * ax11 + x23 * ax21 + x33 * ax31);
        get<2, 3>(*inv) =
            x23 + 0.5 * (x02 * ax03 + x12 * ax13 + x22 * ax23 + x23 * ax33 +
                         x03 * ax02 + x13 * ax12 + x23 * ax22 + x33 * ax32);
        // Phase 3: det = 1/det(Y) via 6 minors (reuse slots 0-6)
        const T& y00 = get<0, 0>(*inv);
        const T& y01 = get<0, 1>(*inv);
        const T& y02 = get<0, 2>(*inv);
        const T& y03 = get<0, 3>(*inv);
        const T& y11 = get<1, 1>(*inv);
        const T& y12 = get<1, 2>(*inv);
        const T& y13 = get<1, 3>(*inv);
        const T& y22 = get<2, 2>(*inv);
        const T& y23 = get<2, 3>(*inv);
        const T& y33 = get<3, 3>(*inv);
        auto& n23_23 = get(get<Tags::TempScalar<0, T>>(buffer));
        auto& n23_13 = get(get<Tags::TempScalar<1, T>>(buffer));
        auto& n23_03 = get(get<Tags::TempScalar<2, T>>(buffer));
        auto& n23_12 = get(get<Tags::TempScalar<3, T>>(buffer));
        auto& n23_02 = get(get<Tags::TempScalar<4, T>>(buffer));
        auto& n23_01 = get(get<Tags::TempScalar<5, T>>(buffer));
        auto& det_y = get(get<Tags::TempScalar<6, T>>(buffer));
        n23_23 = y22 * y33 - y23 * y23;
        n23_13 = y12 * y33 - y23 * y13;
        n23_03 = y02 * y33 - y23 * y03;
        n23_12 = y12 * y23 - y22 * y13;
        n23_02 = y02 * y23 - y22 * y03;
        n23_01 = y02 * y13 - y12 * y03;
        det_y = y00 * (y11 * n23_23 - y12 * n23_13 + y13 * n23_12) -
                y01 * (y01 * n23_23 - y12 * n23_03 + y13 * n23_02) +
                y02 * (y01 * n23_13 - y11 * n23_03 + y13 * n23_01) -
                y03 * (y01 * n23_12 - y11 * n23_02 + y12 * n23_01);
        get(*det) = 1.0 / det_y;
      } else {
        // Analytic: 6 slots, streaming minors.
        // inv[3,3] holds inverse_det = 1/det; multiplied by its cofactor last.
        // Group 1 (m23) → det + inv[0,0], inv[0,1], inv[1,1]; slots reused
        // for Group 2 (m13) → inv[0,2], inv[1,2], inv[2,2]; reused for
        // Group 3 (m12) → inv[0,3], inv[1,3], inv[2,3], inv[3,3].
        TempBuffer<tmpl::list<Tags::TempScalar<0, T>, Tags::TempScalar<1, T>,
                              Tags::TempScalar<2, T>, Tags::TempScalar<3, T>,
                              Tags::TempScalar<4, T>, Tags::TempScalar<5, T>>>
            buffer(get_size(get<0, 0>(tensor)));
        // Group 1: m23 minors → det + inv[0,0], inv[0,1], inv[1,1]
        auto& m23_23 = get(get<Tags::TempScalar<0, T>>(buffer));
        auto& m23_13 = get(get<Tags::TempScalar<1, T>>(buffer));
        auto& m23_03 = get(get<Tags::TempScalar<2, T>>(buffer));
        auto& m23_12 = get(get<Tags::TempScalar<3, T>>(buffer));
        auto& m23_02 = get(get<Tags::TempScalar<4, T>>(buffer));
        auto& m23_01 = get(get<Tags::TempScalar<5, T>>(buffer));
        m23_23 = t22 * t33 - t23 * t23;
        m23_13 = t12 * t33 - t23 * t13;
        m23_03 = t02 * t33 - t23 * t03;
        m23_12 = t12 * t23 - t22 * t13;
        m23_02 = t02 * t23 - t22 * t03;
        m23_01 = t02 * t13 - t12 * t03;
        get(*det) = t00 * (t11 * m23_23 - t12 * m23_13 + t13 * m23_12) -
                    t01 * (t01 * m23_23 - t12 * m23_03 + t13 * m23_02) +
                    t02 * (t01 * m23_13 - t11 * m23_03 + t13 * m23_01) -
                    t03 * (t01 * m23_12 - t11 * m23_02 + t12 * m23_01);
        get<3, 3>(*inv) = 1.0 / get(*det);  // inverse_det; set last
        get<0, 0>(*inv) =
            (t11 * m23_23 - t12 * m23_13 + t13 * m23_12) * get<3, 3>(*inv);
        get<0, 1>(*inv) =
            -(t01 * m23_23 - t02 * m23_13 + t03 * m23_12) * get<3, 3>(*inv);
        get<1, 1>(*inv) =
            (t00 * m23_23 - t02 * m23_03 + t03 * m23_02) * get<3, 3>(*inv);
        // Group 2: reuse same 6 slots for m13 minors → inv[0,2], inv[1,2],
        // inv[2,2]
        auto& m13_23 = get(get<Tags::TempScalar<0, T>>(buffer));
        auto& m13_13 = get(get<Tags::TempScalar<1, T>>(buffer));
        auto& m13_03 = get(get<Tags::TempScalar<2, T>>(buffer));
        auto& m13_12 = get(get<Tags::TempScalar<3, T>>(buffer));
        auto& m13_02 = get(get<Tags::TempScalar<4, T>>(buffer));
        auto& m13_01 = get(get<Tags::TempScalar<5, T>>(buffer));
        m13_23 = t12 * t33 - t13 * t23;
        m13_13 = t11 * t33 - t13 * t13;
        m13_03 = t01 * t33 - t13 * t03;
        m13_12 = t11 * t23 - t12 * t13;
        m13_02 = t01 * t23 - t12 * t03;
        m13_01 = t01 * t13 - t11 * t03;
        get<0, 2>(*inv) =
            (t01 * m13_23 - t02 * m13_13 + t03 * m13_12) * get<3, 3>(*inv);
        get<1, 2>(*inv) =
            -(t00 * m13_23 - t02 * m13_03 + t03 * m13_02) * get<3, 3>(*inv);
        get<2, 2>(*inv) =
            (t00 * m13_13 - t01 * m13_03 + t03 * m13_01) * get<3, 3>(*inv);
        // Group 3: reuse same 6 slots for m12 minors → inv[0,3]..inv[3,3]
        auto& m12_23 = get(get<Tags::TempScalar<0, T>>(buffer));
        auto& m12_13 = get(get<Tags::TempScalar<1, T>>(buffer));
        auto& m12_03 = get(get<Tags::TempScalar<2, T>>(buffer));
        auto& m12_12 = get(get<Tags::TempScalar<3, T>>(buffer));
        auto& m12_02 = get(get<Tags::TempScalar<4, T>>(buffer));
        auto& m12_01 = get(get<Tags::TempScalar<5, T>>(buffer));
        m12_23 = t12 * t23 - t13 * t22;
        m12_13 = t11 * t23 - t13 * t12;
        m12_03 = t01 * t23 - t13 * t02;
        m12_12 = t11 * t22 - t12 * t12;
        m12_02 = t01 * t22 - t12 * t02;
        m12_01 = t01 * t12 - t11 * t02;
        get<0, 3>(*inv) =
            -(t01 * m12_23 - t02 * m12_13 + t03 * m12_12) * get<3, 3>(*inv);
        get<1, 3>(*inv) =
            (t00 * m12_23 - t02 * m12_03 + t03 * m12_02) * get<3, 3>(*inv);
        get<2, 3>(*inv) =
            -(t00 * m12_13 - t01 * m12_03 + t03 * m12_01) * get<3, 3>(*inv);
        get<3, 3>(*inv) *=
            (t00 * m12_12 - t01 * m12_02 + t02 * m12_01);  // set LAST
      }
    }
  }
};
}  // namespace determinant_and_inverse_detail

/// @{
/*!
 * \ingroup TensorGroup
 * \brief Computes the determinant and inverse of a rank-2 Tensor.
 *
 * Computes the determinant and inverse together, because this leads to
 * fewer operations compared to computing the determinant independently.
 *
 * \details
 * Treats the input rank-2 tensor as a matrix. The first (second) index
 * of the tensor corresponds to the rows (columns) of the matrix. The
 * determinant is a scalar tensor. The inverse is a rank-2 tensor whose
 * indices are reversed and of opposite valence relative to the input
 * tensor, i.e. given \f$T_a^b\f$ returns \f$(Tinv)_b^a\f$.
 *
 * The `Method` template parameter selects the algorithm; see
 * `InversionMethod` for the accuracy and performance tradeoffs of each
 * option. The default `Analytic` method uses direct Cramer's rule for all
 * dimensions and is suitable for most use cases.
 *
 * \note
 * When inverting a 4x4 spacetime metric, it is typically more efficient
 * to use the 3+1 decomposition of the 4-metric in terms of lapse,
 * shift, and spatial 3-metric, in which only the spatial 3-metric needs
 * to be inverted.
 */
template <InversionMethod Method = InversionMethod::Analytic, typename T,
          typename Symm, typename Index0, typename Index1>
void determinant_and_inverse(
    const gsl::not_null<Scalar<T>*> det,
    const gsl::not_null<Tensor<
        T, Symm,
        tmpl::list<change_index_up_lo<Index1>, change_index_up_lo<Index0>>>*>
        inv,
    const Tensor<T, Symm, tmpl::list<Index0, Index1>>& tensor) {
  static_assert(Index0::dim == Index1::dim,
                "Cannot take the inverse of a Tensor whose Indices are not "
                "of the same dimensionality.");
  static_assert(Index0::index_type == Index1::index_type,
                "Taking the inverse of a mixed Spatial and Spacetime index "
                "Tensor is not allowed since it's not clear what that means.");
  static_assert(not std::is_integral_v<T>, "Can't invert a Tensor<int>.");

  set_number_of_grid_points(det, tensor);
  set_number_of_grid_points(inv, tensor);
  determinant_and_inverse_detail::DetAndInverseImpl<
      Symm, Index0, Index1>::template apply<Method>(det, inv, tensor);
}

template <InversionMethod Method = InversionMethod::Analytic, typename T,
          typename Symm, typename Index0, typename Index1>
auto determinant_and_inverse(
    const Tensor<T, Symm, tmpl::list<Index0, Index1>>& tensor)
    -> std::pair<Scalar<T>, Tensor<T, Symm,
                                   tmpl::list<change_index_up_lo<Index1>,
                                              change_index_up_lo<Index0>>>> {
  std::pair<Scalar<T>, Tensor<T, Symm,
                              tmpl::list<change_index_up_lo<Index1>,
                                         change_index_up_lo<Index0>>>>
      result{};
  determinant_and_inverse_detail::DetAndInverseImpl<Symm, Index0, Index1>::
      template apply<Method>(make_not_null(&result.first),
                             make_not_null(&result.second), tensor);
  return result;
}
/// @}

/// @{
/*!
 * \ingroup TensorGroup
 * \brief Computes the determinant and inverse of a rank-2 Tensor.
 *
 * Computes the determinant and inverse together, because this leads to fewer
 * operations compared to computing the determinant independently.
 *
 * \tparam DetTag the Tag for the determinant of input Tensor.
 * \tparam InvTag the Tag for the inverse of input Tensor.
 *
 * \details
 * See determinant_and_inverse().
 */
template <typename DetTag, typename InvTag,
          InversionMethod Method = InversionMethod::Analytic, typename T,
          typename Symm, typename Index0, typename Index1>
void determinant_and_inverse(
    const gsl::not_null<Variables<tmpl::list<DetTag, InvTag>>*> det_and_inv,
    const Tensor<T, Symm, tmpl::list<Index0, Index1>>& tensor) {
  static_assert(std::is_same_v<typename DetTag::type, Scalar<T>>,
                "Type of first return tag must correspond to that of input's "
                "determinant.");
  static_assert(
      std::is_same_v<typename InvTag::type,
                     Tensor<T, Symm,
                            tmpl::list<change_index_up_lo<Index1>,
                                       change_index_up_lo<Index0>>>>,
      "Type of second return tag must correspond to that of input's inverse.");
  const auto number_of_grid_points = get<0, 0>(tensor).size();
  if (UNLIKELY(number_of_grid_points != det_and_inv->number_of_grid_points())) {
    det_and_inv->initialize(number_of_grid_points);
  }
  determinant_and_inverse_detail::DetAndInverseImpl<Symm, Index0, Index1>::
      template apply<Method>(make_not_null(&get<DetTag>(*det_and_inv)),
                             make_not_null(&get<InvTag>(*det_and_inv)), tensor);
}

template <typename DetTag, typename InvTag,
          InversionMethod Method = InversionMethod::Analytic, typename T,
          typename Symm, typename Index0, typename Index1>
auto determinant_and_inverse(
    const Tensor<T, Symm, tmpl::list<Index0, Index1>>& tensor)
    -> Variables<tmpl::list<DetTag, InvTag>> {
  static_assert(std::is_same_v<typename DetTag::type, Scalar<T>>,
                "Type of first return tag must correspond to that of input's "
                "determinant.");
  static_assert(std::is_same_v<typename InvTag::type,
                               Tensor<T, Symm,
                                      tmpl::list<change_index_up_lo<Index1>,
                                                 change_index_up_lo<Index0>>>>,
                "Type of second return tag must correspond to that of input's "
                "inverse.");
  Variables<tmpl::list<DetTag, InvTag>> result(get<0, 0>(tensor).size());
  determinant_and_inverse_detail::DetAndInverseImpl<Symm, Index0, Index1>::
      template apply<Method>(make_not_null(&get<DetTag>(result)),
                             make_not_null(&get<InvTag>(result)), tensor);
  return result;
}
/// @}
