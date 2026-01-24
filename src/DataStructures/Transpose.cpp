// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Transpose.hpp"

#include <type_traits>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "Utilities/GenerateInstantiations.hpp"

namespace {
// We assume matrix points to the start of the sub matrix.
//
// Streaming writes didn't improve things with AVX2 on Zen2
// architecture. For AVX-512 streaming might be more useful, but AVX-512 is
// usually not recommended as of 2023 because the CPUs down clock so much
// that all non-math work also suffers.
template <size_t RowsInBlock, size_t ColumnsInBlock>
void transpose_block(double* __restrict__ matrix_transpose,
                     const double* __restrict__ matrix, int32_t columns,
                     int32_t rows);

#if defined(__AVX512F__)
template <int N>
constexpr __mmask8 make_mask() {
  return static_cast<__mmask8>((1u << N) - 1);
}

template <int M, int N>
void transpose_kernel_8x8(double* __restrict__ matrix_transpose,
                          const double* __restrict__ const matrix,
                          const int32_t columns, const int32_t rows) {
  if constexpr (M == 1) {
    // Note that the _mm512_mask_i64scatter_pd instruction has high latency,
    // but our intention here is very clear and the code is extremely simple
    // compared to other implementations. The compiler should be able to
    // optimize as it sees fit.
    constexpr __mmask8 mask = make_mask<N>();
    const __m512d row0 = _mm512_maskz_loadu_pd(mask, matrix);
    const __m512i base = _mm512_setr_epi64(
        0, N > 1 ? 1 : 0, N > 2 ? 2 : 0, N > 3 ? 3 : 0, N > 4 ? 4 : 0,
        N > 5 ? 5 : 0, N > 6 ? 6 : 0, N > 7 ? 7 : 0);
    const __m512i indices = _mm512_mullo_epi64(base, _mm512_set1_epi64(rows));
    _mm512_mask_i64scatter_pd(matrix_transpose, mask, indices, row0, 8);
  } else if constexpr (M == 2) {
    const __mmask8 load_mask = make_mask<N>();
    const __m512d row0 = _mm512_maskz_loadu_pd(load_mask, matrix + 0 * columns);
    const __m512d row1 = _mm512_maskz_loadu_pd(load_mask, matrix + 1 * columns);

    const __m512d tmp0 = _mm512_unpacklo_pd(row0, row1);
    const __m512d tmp1 = _mm512_unpackhi_pd(row0, row1);

    _mm_storeu_pd(matrix_transpose + 0 * rows, _mm512_castpd512_pd128(tmp0));
    if constexpr (N > 1) {
      _mm_storeu_pd(matrix_transpose + 1 * rows, _mm512_castpd512_pd128(tmp1));
    }
    if constexpr (N > 2) {
      _mm_storeu_pd(matrix_transpose + 2 * rows,
                    _mm512_extractf64x2_pd(tmp0, 1));
    }
    if constexpr (N > 3) {
      _mm_storeu_pd(matrix_transpose + 3 * rows,
                    _mm512_extractf64x2_pd(tmp1, 1));
    }
    if constexpr (N > 4) {
      _mm_storeu_pd(matrix_transpose + 4 * rows,
                    _mm512_extractf64x2_pd(tmp0, 2));
    }
    if constexpr (N > 5) {
      _mm_storeu_pd(matrix_transpose + 5 * rows,
                    _mm512_extractf64x2_pd(tmp1, 2));
    }
    if constexpr (N > 6) {
      _mm_storeu_pd(matrix_transpose + 6 * rows,
                    _mm512_extractf64x2_pd(tmp0, 3));
    }
    if constexpr (N > 7) {
      _mm_storeu_pd(matrix_transpose + 7 * rows,
                    _mm512_extractf64x2_pd(tmp1, 3));
    }
  } else if constexpr (M == 3) {
    constexpr __mmask8 load_mask = make_mask<N>();
    constexpr __mmask8 store_mask = make_mask<M>();
    const __m512d row0 = _mm512_maskz_loadu_pd(load_mask, matrix + 0 * columns);
    const __m512d row1 = _mm512_maskz_loadu_pd(load_mask, matrix + 1 * columns);
    const __m512d row2 = _mm512_maskz_loadu_pd(load_mask, matrix + 2 * columns);

    const __m512d tmp0 = _mm512_unpacklo_pd(row0, row1);
    const __m512d tmp1 = _mm512_unpackhi_pd(row0, row1);

    const __m512i idx0 = _mm512_setr_epi64(0, 1, 8, 0, 0, 0, 0, 0);
    const __m512i idx1 = _mm512_setr_epi64(0, 1, 9, 0, 0, 0, 0, 0);
    const __m512i idx2 = _mm512_setr_epi64(2, 3, 10, 0, 0, 0, 0, 0);
    const __m512i idx3 = _mm512_setr_epi64(2, 3, 11, 0, 0, 0, 0, 0);
    const __m512i idx4 = _mm512_setr_epi64(4, 5, 12, 0, 0, 0, 0, 0);

    _mm512_mask_storeu_pd(matrix_transpose + 0 * rows, store_mask,
                          _mm512_permutex2var_pd(tmp0, idx0, row2));
    if constexpr (N > 1) {
      _mm512_mask_storeu_pd(matrix_transpose + 1 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp1, idx1, row2));
    }
    if constexpr (N > 2) {
      _mm512_mask_storeu_pd(matrix_transpose + 2 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp0, idx2, row2));
    }
    if constexpr (N > 3) {
      _mm512_mask_storeu_pd(matrix_transpose + 3 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp1, idx3, row2));
    }
    if constexpr (N > 4) {
      _mm512_mask_storeu_pd(matrix_transpose + 4 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp0, idx4, row2));
    }
    if constexpr (N > 5) {
      const __m512i idx5 = _mm512_setr_epi64(4, 5, 13, 0, 0, 0, 0, 0);
      _mm512_mask_storeu_pd(matrix_transpose + 5 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp1, idx5, row2));
    }
    if constexpr (N > 6) {
      const __m512i idx6 = _mm512_setr_epi64(6, 7, 14, 0, 0, 0, 0, 0);
      _mm512_mask_storeu_pd(matrix_transpose + 6 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp0, idx6, row2));
    }
    if constexpr (N > 7) {
      const __m512i idx7 = _mm512_setr_epi64(6, 7, 15, 0, 0, 0, 0, 0);
      _mm512_mask_storeu_pd(matrix_transpose + 7 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp1, idx7, row2));
    }
  } else if constexpr (M == 4) {
    const __m512d row0 = _mm512_loadu_pd(matrix + 0 * columns);
    const __m512d row1 = _mm512_loadu_pd(matrix + 1 * columns);
    const __m512d row2 = _mm512_loadu_pd(matrix + 2 * columns);
    const __m512d row3 = _mm512_loadu_pd(matrix + 3 * columns);

    const __m512d tmp0 = _mm512_unpacklo_pd(row0, row1);
    const __m512d tmp1 = _mm512_unpackhi_pd(row0, row1);
    const __m512d tmp2 = _mm512_unpacklo_pd(row2, row3);
    const __m512d tmp3 = _mm512_unpackhi_pd(row2, row3);

    const __m512i idx0 = _mm512_setr_epi64(0, 1, 8, 9, 0, 0, 0, 0);
    const __m512i idx1 = _mm512_setr_epi64(2, 3, 10, 11, 0, 0, 0, 0);
    const __m512i idx2 = _mm512_setr_epi64(4, 5, 12, 13, 0, 0, 0, 0);
    const __m512i idx3 = _mm512_setr_epi64(6, 7, 14, 15, 0, 0, 0, 0);

    constexpr __mmask8 store_mask = make_mask<4>();

    _mm512_mask_storeu_pd(matrix_transpose + 0 * rows, store_mask,
                          _mm512_permutex2var_pd(tmp0, idx0, tmp2));
    if constexpr (N > 1) {
      _mm512_mask_storeu_pd(matrix_transpose + 1 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp1, idx0, tmp3));
    }
    if constexpr (N > 2) {
      _mm512_mask_storeu_pd(matrix_transpose + 2 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp0, idx1, tmp2));
    }
    if constexpr (N > 3) {
      _mm512_mask_storeu_pd(matrix_transpose + 3 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp1, idx1, tmp3));
    }
    if constexpr (N > 4) {
      _mm512_mask_storeu_pd(matrix_transpose + 4 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp0, idx2, tmp2));
    }
    if constexpr (N > 5) {
      _mm512_mask_storeu_pd(matrix_transpose + 5 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp1, idx2, tmp3));
    }
    if constexpr (N > 6) {
      _mm512_mask_storeu_pd(matrix_transpose + 6 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp0, idx3, tmp2));
    }
    if constexpr (N > 7) {
      _mm512_mask_storeu_pd(matrix_transpose + 7 * rows, store_mask,
                            _mm512_permutex2var_pd(tmp1, idx3, tmp3));
    }
  } else if constexpr (M == 5) {
    const __m512d row0 = _mm512_loadu_pd(matrix + 0 * columns);
    const __m512d row1 = _mm512_loadu_pd(matrix + 1 * columns);
    const __m512d row2 = _mm512_loadu_pd(matrix + 2 * columns);
    const __m512d row3 = _mm512_loadu_pd(matrix + 3 * columns);
    const __m512d row4 = _mm512_loadu_pd(matrix + 4 * columns);

    const __m512d tmp0 = _mm512_unpacklo_pd(row0, row1);
    const __m512d tmp1 = _mm512_unpackhi_pd(row0, row1);
    const __m512d tmp2 = _mm512_unpacklo_pd(row2, row3);
    const __m512d tmp3 = _mm512_unpackhi_pd(row2, row3);

    // First gather elements 0,1 from tmp0/tmp2, then element from row4
    const __m512i idx0 = _mm512_setr_epi64(0, 1, 8, 9, 0, 0, 0, 0);
    const __m512i idx1 = _mm512_setr_epi64(2, 3, 10, 11, 0, 0, 0, 0);
    const __m512i idx2 = _mm512_setr_epi64(4, 5, 12, 13, 0, 0, 0, 0);
    const __m512i idx3 = _mm512_setr_epi64(6, 7, 14, 15, 0, 0, 0, 0);

    const __m512d out0_partial = _mm512_permutex2var_pd(tmp0, idx0, tmp2);
    const __m512d out1_partial = _mm512_permutex2var_pd(tmp1, idx0, tmp3);
    const __m512d out2_partial = _mm512_permutex2var_pd(tmp0, idx1, tmp2);
    const __m512d out3_partial = _mm512_permutex2var_pd(tmp1, idx1, tmp3);
    const __m512d out4_partial = _mm512_permutex2var_pd(tmp0, idx2, tmp2);
    const __m512d out5_partial = _mm512_permutex2var_pd(tmp1, idx2, tmp3);
    const __m512d out6_partial = _mm512_permutex2var_pd(tmp0, idx3, tmp2);
    const __m512d out7_partial = _mm512_permutex2var_pd(tmp1, idx3, tmp3);

    // Now insert element from row4 at position 4
    const __m512i idx_insert0 = _mm512_setr_epi64(0, 1, 2, 3, 8, 0, 0, 0);
    const __m512i idx_insert1 = _mm512_setr_epi64(0, 1, 2, 3, 9, 0, 0, 0);
    const __m512i idx_insert2 = _mm512_setr_epi64(0, 1, 2, 3, 10, 0, 0, 0);
    const __m512i idx_insert3 = _mm512_setr_epi64(0, 1, 2, 3, 11, 0, 0, 0);
    const __m512i idx_insert4 = _mm512_setr_epi64(0, 1, 2, 3, 12, 0, 0, 0);
    const __m512i idx_insert5 = _mm512_setr_epi64(0, 1, 2, 3, 13, 0, 0, 0);
    const __m512i idx_insert6 = _mm512_setr_epi64(0, 1, 2, 3, 14, 0, 0, 0);
    const __m512i idx_insert7 = _mm512_setr_epi64(0, 1, 2, 3, 15, 0, 0, 0);

    constexpr __mmask8 store_mask = make_mask<5>();

    _mm512_mask_storeu_pd(
        matrix_transpose + 0 * rows, store_mask,
        _mm512_permutex2var_pd(out0_partial, idx_insert0, row4));
    if constexpr (N > 1) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 1 * rows, store_mask,
          _mm512_permutex2var_pd(out1_partial, idx_insert1, row4));
    }
    if constexpr (N > 2) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 2 * rows, store_mask,
          _mm512_permutex2var_pd(out2_partial, idx_insert2, row4));
    }
    if constexpr (N > 3) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 3 * rows, store_mask,
          _mm512_permutex2var_pd(out3_partial, idx_insert3, row4));
    }
    if constexpr (N > 4) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 4 * rows, store_mask,
          _mm512_permutex2var_pd(out4_partial, idx_insert4, row4));
    }
    if constexpr (N > 5) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 5 * rows, store_mask,
          _mm512_permutex2var_pd(out5_partial, idx_insert5, row4));
    }
    if constexpr (N > 6) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 6 * rows, store_mask,
          _mm512_permutex2var_pd(out6_partial, idx_insert6, row4));
    }
    if constexpr (N > 7) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 7 * rows, store_mask,
          _mm512_permutex2var_pd(out7_partial, idx_insert7, row4));
    }
  } else if constexpr (M == 6) {
    const __m512d row0 = _mm512_loadu_pd(matrix + 0 * columns);
    const __m512d row1 = _mm512_loadu_pd(matrix + 1 * columns);
    const __m512d row2 = _mm512_loadu_pd(matrix + 2 * columns);
    const __m512d row3 = _mm512_loadu_pd(matrix + 3 * columns);
    const __m512d row4 = _mm512_loadu_pd(matrix + 4 * columns);
    const __m512d row5 = _mm512_loadu_pd(matrix + 5 * columns);

    const __m512d tmp0 = _mm512_unpacklo_pd(row0, row1);
    const __m512d tmp1 = _mm512_unpackhi_pd(row0, row1);
    const __m512d tmp2 = _mm512_unpacklo_pd(row2, row3);
    const __m512d tmp3 = _mm512_unpackhi_pd(row2, row3);
    const __m512d tmp4 = _mm512_unpacklo_pd(row4, row5);
    const __m512d tmp5 = _mm512_unpackhi_pd(row4, row5);

    // First combine tmp0 and tmp2 to get elements from rows 0-3
    const __m512i idx01 = _mm512_setr_epi64(0, 1, 8, 9, 0, 0, 0, 0);
    const __m512i idx23 = _mm512_setr_epi64(2, 3, 10, 11, 0, 0, 0, 0);
    const __m512i idx45 = _mm512_setr_epi64(4, 5, 12, 13, 0, 0, 0, 0);
    const __m512i idx67 = _mm512_setr_epi64(6, 7, 14, 15, 0, 0, 0, 0);

    const __m512d out0_partial = _mm512_permutex2var_pd(tmp0, idx01, tmp2);
    const __m512d out1_partial = _mm512_permutex2var_pd(tmp1, idx01, tmp3);
    const __m512d out2_partial = _mm512_permutex2var_pd(tmp0, idx23, tmp2);
    const __m512d out3_partial = _mm512_permutex2var_pd(tmp1, idx23, tmp3);
    const __m512d out4_partial = _mm512_permutex2var_pd(tmp0, idx45, tmp2);
    const __m512d out5_partial = _mm512_permutex2var_pd(tmp1, idx45, tmp3);
    const __m512d out6_partial = _mm512_permutex2var_pd(tmp0, idx67, tmp2);
    const __m512d out7_partial = _mm512_permutex2var_pd(tmp1, idx67, tmp3);

    // Now insert elements from tmp4/tmp5 (rows 4-5) at positions 4-5
    const __m512i idx_insert0 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 0, 0);
    const __m512i idx_insert1 = _mm512_setr_epi64(0, 1, 2, 3, 10, 11, 0, 0);
    const __m512i idx_insert2 = _mm512_setr_epi64(0, 1, 2, 3, 12, 13, 0, 0);
    const __m512i idx_insert3 = _mm512_setr_epi64(0, 1, 2, 3, 14, 15, 0, 0);

    constexpr __mmask8 store_mask = make_mask<6>();

    _mm512_mask_storeu_pd(
        matrix_transpose + 0 * rows, store_mask,
        _mm512_permutex2var_pd(out0_partial, idx_insert0, tmp4));
    if constexpr (N > 1) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 1 * rows, store_mask,
          _mm512_permutex2var_pd(out1_partial, idx_insert0, tmp5));
    }
    if constexpr (N > 2) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 2 * rows, store_mask,
          _mm512_permutex2var_pd(out2_partial, idx_insert1, tmp4));
    }
    if constexpr (N > 3) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 3 * rows, store_mask,
          _mm512_permutex2var_pd(out3_partial, idx_insert1, tmp5));
    }
    if constexpr (N > 4) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 4 * rows, store_mask,
          _mm512_permutex2var_pd(out4_partial, idx_insert2, tmp4));
    }
    if constexpr (N > 5) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 5 * rows, store_mask,
          _mm512_permutex2var_pd(out5_partial, idx_insert2, tmp5));
    }
    if constexpr (N > 6) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 6 * rows, store_mask,
          _mm512_permutex2var_pd(out6_partial, idx_insert3, tmp4));
    }
    if constexpr (N > 7) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 7 * rows, store_mask,
          _mm512_permutex2var_pd(out7_partial, idx_insert3, tmp5));
    }
  } else if constexpr (M == 7) {
    const __m512d row0 = _mm512_loadu_pd(matrix + 0 * columns);
    const __m512d row1 = _mm512_loadu_pd(matrix + 1 * columns);
    const __m512d row2 = _mm512_loadu_pd(matrix + 2 * columns);
    const __m512d row3 = _mm512_loadu_pd(matrix + 3 * columns);
    const __m512d row4 = _mm512_loadu_pd(matrix + 4 * columns);
    const __m512d row5 = _mm512_loadu_pd(matrix + 5 * columns);
    const __m512d row6 = _mm512_loadu_pd(matrix + 6 * columns);

    const __m512d tmp0 = _mm512_unpacklo_pd(row0, row1);
    const __m512d tmp1 = _mm512_unpackhi_pd(row0, row1);
    const __m512d tmp2 = _mm512_unpacklo_pd(row2, row3);
    const __m512d tmp3 = _mm512_unpackhi_pd(row2, row3);
    const __m512d tmp4 = _mm512_unpacklo_pd(row4, row5);
    const __m512d tmp5 = _mm512_unpackhi_pd(row4, row5);

    // First combine tmp0 and tmp2 to get elements from rows 0-3
    const __m512i idx01 = _mm512_setr_epi64(0, 1, 8, 9, 0, 0, 0, 0);
    const __m512i idx23 = _mm512_setr_epi64(2, 3, 10, 11, 0, 0, 0, 0);
    const __m512i idx45 = _mm512_setr_epi64(4, 5, 12, 13, 0, 0, 0, 0);
    const __m512i idx67 = _mm512_setr_epi64(6, 7, 14, 15, 0, 0, 0, 0);

    const __m512d out0_partial = _mm512_permutex2var_pd(tmp0, idx01, tmp2);
    const __m512d out1_partial = _mm512_permutex2var_pd(tmp1, idx01, tmp3);
    const __m512d out2_partial = _mm512_permutex2var_pd(tmp0, idx23, tmp2);
    const __m512d out3_partial = _mm512_permutex2var_pd(tmp1, idx23, tmp3);
    const __m512d out4_partial = _mm512_permutex2var_pd(tmp0, idx45, tmp2);
    const __m512d out5_partial = _mm512_permutex2var_pd(tmp1, idx45, tmp3);
    const __m512d out6_partial = _mm512_permutex2var_pd(tmp0, idx67, tmp2);
    const __m512d out7_partial = _mm512_permutex2var_pd(tmp1, idx67, tmp3);

    // Insert elements from tmp4/tmp5 (rows 4-5) at positions 4-5
    const __m512i idx_insert01 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 0, 0);
    const __m512i idx_insert23 = _mm512_setr_epi64(0, 1, 2, 3, 10, 11, 0, 0);
    const __m512i idx_insert45 = _mm512_setr_epi64(0, 1, 2, 3, 12, 13, 0, 0);
    const __m512i idx_insert67 = _mm512_setr_epi64(0, 1, 2, 3, 14, 15, 0, 0);

    const __m512d out0_partial2 =
        _mm512_permutex2var_pd(out0_partial, idx_insert01, tmp4);
    const __m512d out1_partial2 =
        _mm512_permutex2var_pd(out1_partial, idx_insert01, tmp5);
    const __m512d out2_partial2 =
        _mm512_permutex2var_pd(out2_partial, idx_insert23, tmp4);
    const __m512d out3_partial2 =
        _mm512_permutex2var_pd(out3_partial, idx_insert23, tmp5);
    const __m512d out4_partial2 =
        _mm512_permutex2var_pd(out4_partial, idx_insert45, tmp4);
    const __m512d out5_partial2 =
        _mm512_permutex2var_pd(out5_partial, idx_insert45, tmp5);
    const __m512d out6_partial2 =
        _mm512_permutex2var_pd(out6_partial, idx_insert67, tmp4);
    const __m512d out7_partial2 =
        _mm512_permutex2var_pd(out7_partial, idx_insert67, tmp5);

    // Insert element from row6 at position 6
    const __m512i idx_final0 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 8, 0);
    const __m512i idx_final1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 9, 0);
    const __m512i idx_final2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 10, 0);
    const __m512i idx_final3 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 11, 0);
    const __m512i idx_final4 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 12, 0);
    const __m512i idx_final5 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 13, 0);
    const __m512i idx_final6 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 14, 0);
    const __m512i idx_final7 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 15, 0);

    constexpr __mmask8 store_mask = make_mask<7>();

    _mm512_mask_storeu_pd(
        matrix_transpose + 0 * rows, store_mask,
        _mm512_permutex2var_pd(out0_partial2, idx_final0, row6));
    if constexpr (N > 1) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 1 * rows, store_mask,
          _mm512_permutex2var_pd(out1_partial2, idx_final1, row6));
    }
    if constexpr (N > 2) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 2 * rows, store_mask,
          _mm512_permutex2var_pd(out2_partial2, idx_final2, row6));
    }
    if constexpr (N > 3) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 3 * rows, store_mask,
          _mm512_permutex2var_pd(out3_partial2, idx_final3, row6));
    }
    if constexpr (N > 4) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 4 * rows, store_mask,
          _mm512_permutex2var_pd(out4_partial2, idx_final4, row6));
    }
    if constexpr (N > 5) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 5 * rows, store_mask,
          _mm512_permutex2var_pd(out5_partial2, idx_final5, row6));
    }
    if constexpr (N > 6) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 6 * rows, store_mask,
          _mm512_permutex2var_pd(out6_partial2, idx_final6, row6));
    }
    if constexpr (N > 7) {
      _mm512_mask_storeu_pd(
          matrix_transpose + 7 * rows, store_mask,
          _mm512_permutex2var_pd(out7_partial2, idx_final7, row6));
    }
  } else if constexpr (M == 8) {
    const __m512d row0 = _mm512_loadu_pd(matrix + 0 * columns);
    const __m512d row1 = _mm512_loadu_pd(matrix + 1 * columns);
    const __m512d row2 = _mm512_loadu_pd(matrix + 2 * columns);
    const __m512d row3 = _mm512_loadu_pd(matrix + 3 * columns);
    const __m512d row4 = _mm512_loadu_pd(matrix + 4 * columns);
    const __m512d row5 = _mm512_loadu_pd(matrix + 5 * columns);
    const __m512d row6 = _mm512_loadu_pd(matrix + 6 * columns);
    const __m512d row7 = _mm512_loadu_pd(matrix + 7 * columns);

    const __m512d tmp0 = _mm512_unpacklo_pd(row0, row1);
    const __m512d tmp1 = _mm512_unpackhi_pd(row0, row1);
    const __m512d tmp2 = _mm512_unpacklo_pd(row2, row3);
    const __m512d tmp3 = _mm512_unpackhi_pd(row2, row3);
    const __m512d tmp4 = _mm512_unpacklo_pd(row4, row5);
    const __m512d tmp5 = _mm512_unpackhi_pd(row4, row5);
    const __m512d tmp6 = _mm512_unpacklo_pd(row6, row7);
    const __m512d tmp7 = _mm512_unpackhi_pd(row6, row7);

    const __m512i idx01 = _mm512_setr_epi64(0, 1, 8, 9, 0, 0, 0, 0);
    const __m512i idx23 = _mm512_setr_epi64(2, 3, 10, 11, 0, 0, 0, 0);
    const __m512i idx45 = _mm512_setr_epi64(4, 5, 12, 13, 0, 0, 0, 0);
    const __m512i idx67 = _mm512_setr_epi64(6, 7, 14, 15, 0, 0, 0, 0);

    const __m512d out0_partial = _mm512_permutex2var_pd(tmp0, idx01, tmp2);
    const __m512d out1_partial = _mm512_permutex2var_pd(tmp1, idx01, tmp3);
    const __m512d out2_partial = _mm512_permutex2var_pd(tmp0, idx23, tmp2);
    const __m512d out3_partial = _mm512_permutex2var_pd(tmp1, idx23, tmp3);
    const __m512d out4_partial = _mm512_permutex2var_pd(tmp0, idx45, tmp2);
    const __m512d out5_partial = _mm512_permutex2var_pd(tmp1, idx45, tmp3);
    const __m512d out6_partial = _mm512_permutex2var_pd(tmp0, idx67, tmp2);
    const __m512d out7_partial = _mm512_permutex2var_pd(tmp1, idx67, tmp3);

    const __m512d out0_partial2 = _mm512_permutex2var_pd(tmp4, idx01, tmp6);
    const __m512d out1_partial2 = _mm512_permutex2var_pd(tmp5, idx01, tmp7);
    const __m512d out2_partial2 = _mm512_permutex2var_pd(tmp4, idx23, tmp6);
    const __m512d out3_partial2 = _mm512_permutex2var_pd(tmp5, idx23, tmp7);
    const __m512d out4_partial2 = _mm512_permutex2var_pd(tmp4, idx45, tmp6);
    const __m512d out5_partial2 = _mm512_permutex2var_pd(tmp5, idx45, tmp7);
    const __m512d out6_partial2 = _mm512_permutex2var_pd(tmp4, idx67, tmp6);
    const __m512d out7_partial2 = _mm512_permutex2var_pd(tmp5, idx67, tmp7);

    const __m512i idx_final = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);

    _mm512_storeu_pd(
        matrix_transpose + 0 * rows,
        _mm512_permutex2var_pd(out0_partial, idx_final, out0_partial2));
    if constexpr (N > 1) {
      _mm512_storeu_pd(
          matrix_transpose + 1 * rows,
          _mm512_permutex2var_pd(out1_partial, idx_final, out1_partial2));
    }
    if constexpr (N > 2) {
      _mm512_storeu_pd(
          matrix_transpose + 2 * rows,
          _mm512_permutex2var_pd(out2_partial, idx_final, out2_partial2));
    }
    if constexpr (N > 3) {
      _mm512_storeu_pd(
          matrix_transpose + 3 * rows,
          _mm512_permutex2var_pd(out3_partial, idx_final, out3_partial2));
    }
    if constexpr (N > 4) {
      _mm512_storeu_pd(
          matrix_transpose + 4 * rows,
          _mm512_permutex2var_pd(out4_partial, idx_final, out4_partial2));
    }
    if constexpr (N > 5) {
      _mm512_storeu_pd(
          matrix_transpose + 5 * rows,
          _mm512_permutex2var_pd(out5_partial, idx_final, out5_partial2));
    }
    if constexpr (N > 6) {
      _mm512_storeu_pd(
          matrix_transpose + 6 * rows,
          _mm512_permutex2var_pd(out6_partial, idx_final, out6_partial2));
    }
    if constexpr (N > 7) {
      _mm512_storeu_pd(
          matrix_transpose + 7 * rows,
          _mm512_permutex2var_pd(out7_partial, idx_final, out7_partial2));
    }
  }
}

#define GET_M(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_N(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                               \
  template <>                                                                \
  void transpose_block<GET_M(data), GET_N(data)>(                            \
      double* __restrict__ matrix_transpose,                                 \
      const double* __restrict__ const matrix, const int32_t columns,        \
      const int32_t rows) {                                                  \
    transpose_kernel_8x8<GET_M(data), GET_N(data)>(matrix_transpose, matrix, \
                                                   columns, rows);           \
  }

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3, 4), (5, 6, 7, 8))
GENERATE_INSTANTIATIONS(INSTANTIATION, (5, 6, 7, 8), (1, 2, 3, 4))
GENERATE_INSTANTIATIONS(INSTANTIATION, (5, 6, 7, 8), (5, 6, 7, 8))

#undef INSTANTIATION
#undef GET_N
#undef GET_M
#endif

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
#if defined(__AVX__)
template <>
void transpose_block<4, 4>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256d row0 = _mm256_loadu_pd(matrix + 0 * columns);
  const __m256d row1 = _mm256_loadu_pd(matrix + 1 * columns);
  const __m256d row2 = _mm256_loadu_pd(matrix + 2 * columns);
  const __m256d row3 = _mm256_loadu_pd(matrix + 3 * columns);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp2 = _mm256_shuffle_pd((row0), (row1), 0b1111);
  const __m256d tmp1 = _mm256_shuffle_pd((row2), (row3), 0b0000);
  const __m256d tmp3 = _mm256_shuffle_pd((row2), (row3), 0b1111);

  _mm256_storeu_pd(matrix_transpose + 0 * rows,
                   _mm256_permute2f128_pd(tmp0, tmp1, 0x20));
  _mm256_storeu_pd(matrix_transpose + 1 * rows,
                   _mm256_permute2f128_pd(tmp2, tmp3, 0x20));
  _mm256_storeu_pd(matrix_transpose + 2 * rows,
                   _mm256_permute2f128_pd(tmp0, tmp1, 0x31));
  _mm256_storeu_pd(matrix_transpose + 3 * rows,
                   _mm256_permute2f128_pd(tmp2, tmp3, 0x31));
}

template <>
void transpose_block<3, 4>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 = _mm256_loadu_pd(matrix + 0 * columns);
  const __m256d row1 = _mm256_loadu_pd(matrix + 1 * columns);
  const __m256d row2 = _mm256_loadu_pd(matrix + 2 * columns);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp2 = _mm256_shuffle_pd((row0), (row1), 0b1111);
  const __m256d tmp1 = _mm256_shuffle_pd((row2), (row2), 0b0000);
  const __m256d tmp3 = _mm256_shuffle_pd((row2), (row2), 0b1111);

  _mm256_maskstore_pd(matrix_transpose + 0 * rows, mask,
                      _mm256_permute2f128_pd(tmp0, tmp1, 0x20));
  _mm256_maskstore_pd(matrix_transpose + 1 * rows, mask,
                      _mm256_permute2f128_pd(tmp2, tmp3, 0x20));
  _mm256_maskstore_pd(matrix_transpose + 2 * rows, mask,
                      _mm256_permute2f128_pd(tmp0, tmp1, 0x31));
  _mm256_maskstore_pd(matrix_transpose + 3 * rows, mask,
                      _mm256_permute2f128_pd(tmp2, tmp3, 0x31));
}

template <>
void transpose_block<2, 4>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256d row0 = _mm256_loadu_pd(matrix + 0 * columns);
  const __m256d row1 = _mm256_loadu_pd(matrix + 1 * columns);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp1 = _mm256_shuffle_pd((row0), (row1), 0b1111);

  _mm_storeu_pd(matrix_transpose + 0 * rows, _mm256_extractf128_pd(tmp0, 0));
  _mm_storeu_pd(matrix_transpose + 1 * rows, _mm256_extractf128_pd(tmp1, 0));
  _mm_storeu_pd(matrix_transpose + 2 * rows, _mm256_extractf128_pd(tmp0, 1));
  _mm_storeu_pd(matrix_transpose + 3 * rows, _mm256_extractf128_pd(tmp1, 1));
}

template <>
void transpose_block<1, 4>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t /*columns*/, const int32_t rows) {
  const __m256d row0 = _mm256_loadu_pd(matrix);

  const __m256d tmp0 = _mm256_shuffle_pd(row0, row0, 0b0000);
  const __m256d tmp1 = _mm256_shuffle_pd(row0, row0, 0b1111);

  const __m128i store_mask = _mm_set_epi64x(0, -1);
  _mm_maskstore_pd(matrix_transpose + 0 * rows, store_mask,
                   _mm256_castpd256_pd128(tmp0));
  _mm_maskstore_pd(matrix_transpose + 1 * rows, store_mask,
                   _mm256_castpd256_pd128(tmp1));
  _mm_maskstore_pd(matrix_transpose + 2 * rows, store_mask,
                   _mm256_extractf128_pd(tmp0, 1));
  _mm_maskstore_pd(matrix_transpose + 3 * rows, store_mask,
                   _mm256_extractf128_pd(tmp1, 1));
}

template <>
void transpose_block<4, 3>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 = _mm256_maskload_pd(matrix + 0 * columns, mask);
  const __m256d row1 = _mm256_maskload_pd(matrix + 1 * columns, mask);
  const __m256d row2 = _mm256_maskload_pd(matrix + 2 * columns, mask);
  const __m256d row3 = _mm256_maskload_pd(matrix + 3 * columns, mask);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp2 = _mm256_shuffle_pd((row0), (row1), 0b1111);
  const __m256d tmp1 = _mm256_shuffle_pd((row2), (row3), 0b0000);
  const __m256d tmp3 = _mm256_shuffle_pd((row2), (row3), 0b1111);

  _mm256_storeu_pd(matrix_transpose + 0 * rows, _mm256_permute2f128_pd(tmp0, tmp1, 0x20));
  _mm256_storeu_pd(matrix_transpose + 1 * rows,
                   _mm256_permute2f128_pd(tmp2, tmp3, 0x20));
  _mm256_storeu_pd(matrix_transpose + 2 * rows,
                   _mm256_permute2f128_pd(tmp0, tmp1, 0x31));
}

template <>
void transpose_block<4, 2>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256d rows0_1 = _mm256_permute2f128_pd(
      _mm256_castpd128_pd256(_mm_loadu_pd(matrix + 0 * columns)),
      _mm256_castpd128_pd256(_mm_loadu_pd(matrix + 2 * columns)), 0b00100000);
  const __m256d rows2_3 = _mm256_permute2f128_pd(
      _mm256_castpd128_pd256(_mm_loadu_pd(matrix + 1 * columns)),
      _mm256_castpd128_pd256(_mm_loadu_pd(matrix + 3 * columns)), 0b00100000);

  _mm256_storeu_pd(matrix_transpose + 0 * rows,
                   _mm256_unpacklo_pd(rows0_1, rows2_3));
  _mm256_storeu_pd(matrix_transpose + 1 * rows,
                   _mm256_unpackhi_pd(rows0_1, rows2_3));
}

template <>
void transpose_block<4, 1>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  // We load the 4 rows into SSE registers, and then combine them into a
  // single AVX register for write.
  const __m128d row0 = _mm_load_pd1(matrix + 0 * columns);
  const __m128d row1 = _mm_load_pd1(matrix + 1 * columns);
  const __m128d row2 = _mm_load_pd1(matrix + 2 * columns);
  const __m128d row3 = _mm_load_pd1(matrix + 3 * columns);

  _mm256_storeu_pd(matrix_transpose + 0 * rows,
                   _mm256_insertf128_pd(
                       _mm256_castpd128_pd256(_mm_shuffle_pd(row0, row1, 0b00)),
                       _mm_shuffle_pd(row2, row3, 0b00), 1));
}

template <>
void transpose_block<3, 3>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 = _mm256_maskload_pd(matrix + 0 * columns, mask);
  const __m256d row1 = _mm256_maskload_pd(matrix + 1 * columns, mask);
  const __m256d row2 = _mm256_maskload_pd(matrix + 2 * columns, mask);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp2 = _mm256_shuffle_pd((row0), (row1), 0b1111);
  const __m256d tmp1 = _mm256_shuffle_pd((row2), (row2), 0b0000);
  const __m256d tmp3 = _mm256_shuffle_pd((row2), (row2), 0b1111);

  _mm256_maskstore_pd(matrix_transpose + 0 * rows, mask,
                      _mm256_permute2f128_pd(tmp0, tmp1, 0x20));
  _mm256_maskstore_pd(matrix_transpose + 1 * rows, mask,
                      _mm256_permute2f128_pd(tmp2, tmp3, 0x20));
  _mm256_maskstore_pd(matrix_transpose + 2 * rows, mask,
                      _mm256_permute2f128_pd(tmp0, tmp1, 0x31));
}

template <>
void transpose_block<3, 2>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 =
      _mm256_castpd128_pd256(_mm_loadu_pd(matrix + 0 * columns));
  const __m256d row1 =
      _mm256_castpd128_pd256(_mm_loadu_pd(matrix + 1 * columns));
  const __m256d row2 =
      _mm256_castpd128_pd256(_mm_loadu_pd(matrix + 2 * columns));

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp2 = _mm256_shuffle_pd((row0), (row1), 0b1111);
  const __m256d tmp1 = _mm256_shuffle_pd((row2), (row2), 0b0000);
  const __m256d tmp3 = _mm256_shuffle_pd((row2), (row2), 0b1111);

  _mm256_maskstore_pd(matrix_transpose + 0 * rows, mask,
                      _mm256_permute2f128_pd(tmp0, tmp1, 0x20));
  _mm256_maskstore_pd(matrix_transpose + 1 * rows, mask,
                      _mm256_permute2f128_pd(tmp2, tmp3, 0x20));
}

template <>
void transpose_block<3, 1>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  // We load the 3 rows into SSE registers, and then combine them into a
  // single AVX register for write.
  const __m128d row0 = _mm_load_pd1(matrix + 0 * columns);
  const __m128d row1 = _mm_load_pd1(matrix + 1 * columns);
  const __m128d row2 = _mm_load_pd1(matrix + 2 * columns);

  _mm256_maskstore_pd(
      matrix_transpose + 0 * rows, mask,
      _mm256_insertf128_pd(
          _mm256_castpd128_pd256(_mm_shuffle_pd(row0, row1, 0b00)), row2, 1));
}

template <>
void transpose_block<2, 3>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 = _mm256_maskload_pd(matrix + 0 * columns, mask);
  const __m256d row1 = _mm256_maskload_pd(matrix + 1 * columns, mask);

  const __m256d tmp0 = _mm256_shuffle_pd((row0), (row1), 0b0000);
  const __m256d tmp1 = _mm256_shuffle_pd((row0), (row1), 0b1111);

  _mm_storeu_pd(matrix_transpose + 0 * rows, _mm256_castpd256_pd128(tmp0));
  _mm_storeu_pd(matrix_transpose + 1 * rows, _mm256_castpd256_pd128(tmp1));
  _mm_storeu_pd(matrix_transpose + 2 * rows, _mm256_extractf128_pd(tmp0, 1));
}

template <>
void transpose_block<1, 3>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
  const __m256d row0 = _mm256_maskload_pd(matrix + 0 * columns, mask);

  const __m256d tmp0 = _mm256_shuffle_pd(row0, row0, 0b0000);
  const __m256d tmp1 = _mm256_shuffle_pd(row0, row0, 0b1111);

  const __m128i store_mask = _mm_set_epi64x(0, -1);
  _mm_maskstore_pd(matrix_transpose + 0 * rows, store_mask,
                   _mm256_castpd256_pd128(tmp0));
  _mm_maskstore_pd(matrix_transpose + 1 * rows, store_mask,
                   _mm256_castpd256_pd128(tmp1));
  _mm_maskstore_pd(matrix_transpose + 2 * rows, store_mask,
                   _mm256_extractf128_pd(tmp0, 1));
}
#endif

#if defined(__SSE2__)
template <>
void transpose_block<2, 2>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t rows) {
  const __m128d row0 = _mm_loadu_pd(matrix);
  const __m128d row1 = _mm_loadu_pd(matrix + columns);

  const __m128d tmp0 = _mm_shuffle_pd(row0, row1, 0b00);
  const __m128d tmp1 = _mm_shuffle_pd(row0, row1, 0b11);

  _mm_storeu_pd(matrix_transpose, tmp0);
  _mm_storeu_pd(matrix_transpose + rows, tmp1);
}

template <>
void transpose_block<2, 1>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t columns, const int32_t /*rows*/) {
  const __m128d row0 = _mm_load_pd1(matrix + 0 * columns);
  const __m128d row1 = _mm_load_pd1(matrix + 1 * columns);

  const __m128d tmp0 = _mm_shuffle_pd(row0, row1, 0b00);

  _mm_storeu_pd(matrix_transpose, tmp0);
}

template <>
void transpose_block<1, 2>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t /*columns*/, const int32_t rows) {
#if defined(__AVX__)
  const __m128d row = _mm_loadu_pd(matrix);
  _mm_maskstore_pd(matrix_transpose + 0 * rows, _mm_set_epi64x(0, -1), row);
  _mm_maskstore_pd(matrix_transpose + 1 * rows - 1, _mm_set_epi64x(-1, 0), row);
#else
  matrix_transpose[0] = matrix[0];
  matrix_transpose[rows] = matrix[1];
#endif
}
#endif
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

template <>
void transpose_block<1, 1>(double* __restrict__ matrix_transpose,
                           const double* __restrict__ const matrix,
                           const int32_t /*columns*/, const int32_t /*rows*/) {
  *matrix_transpose = *matrix;
}

template <int32_t BlockSize, int32_t RowExcess, int32_t ColumnExcess>
void transpose_impl(double* __restrict__ matrix_transpose,  //
                    const double* __restrict__ const matrix,
                    const int32_t in_number_of_rows,
                    const int32_t in_number_of_columns) {
  const int32_t bound_on_rows = in_number_of_rows - RowExcess;
  const int32_t bound_on_columns = in_number_of_columns - ColumnExcess;

  for (int32_t row_index = 0UL; row_index < bound_on_rows;
       row_index += BlockSize) {
    for (int32_t column_index = 0UL; column_index < bound_on_columns;
         column_index += BlockSize) {
      if constexpr (BlockSize != 1) {
        transpose_block<BlockSize, BlockSize>(
            matrix_transpose + row_index + in_number_of_rows * column_index,
            matrix + column_index + in_number_of_columns * row_index,
            in_number_of_columns, in_number_of_rows);
      } else {
        static_assert(BlockSize == 1);
        static_assert(RowExcess == 0);
        static_assert(ColumnExcess == 0);
        transpose_block<1, 1>(
            matrix_transpose + row_index + in_number_of_rows * column_index,
            matrix + column_index + in_number_of_columns * row_index,
            in_number_of_columns, in_number_of_rows);
      }
    }
    // Handle remainder in row, that is, deal with extra columns.
    if constexpr (BlockSize > 1 and ColumnExcess != 0) {
      const int32_t column_index = bound_on_columns;
      transpose_block<BlockSize, ColumnExcess>(
          matrix_transpose + row_index + in_number_of_rows * column_index,
          matrix + column_index + in_number_of_columns * row_index,
          in_number_of_columns, in_number_of_rows);
    }
  }

  // Now deal with excess in either the columns or rows.
  //
  // We have the choice of either having the extra loops of the inner index
  // (currently row_index)  inside the main loop above or down below. This is a
  // tradeoff between data cache and instruction cache.
  if constexpr (BlockSize > 1 and RowExcess != 0) {
    const int32_t row_index = bound_on_rows;
    for (int32_t column_index = 0UL; column_index < bound_on_columns;
         column_index += BlockSize) {
      transpose_block<RowExcess, BlockSize>(
          matrix_transpose + row_index + in_number_of_rows * column_index,
          matrix + column_index + in_number_of_columns * row_index,
          in_number_of_columns, in_number_of_rows);
    }
    if constexpr (ColumnExcess != 0) {
      const int32_t column_index = bound_on_columns;
      transpose_block<RowExcess, ColumnExcess>(
          matrix_transpose + row_index + in_number_of_rows * column_index,
          matrix + column_index + in_number_of_columns * row_index,
          in_number_of_columns, in_number_of_rows);
    }
  }
}
}  // namespace

namespace detail {
void transpose_impl(double* matrix_transpose, const double* const matrix,
                    const int32_t number_of_rows,
                    const int32_t number_of_columns) {
  constexpr size_t block_size =
#if defined(__AVX512F__)
      8
#elif defined(__AVX__)
      4
#elif defined(__SSE2__)
      2
#else
      1
#endif
      ;
  const auto forward_to_impl = [&](auto row_excess_v) {
    constexpr size_t row_excess = decltype(row_excess_v)::value;
    switch (number_of_columns % static_cast<int32_t>(block_size)) {
#if defined(__AVX512F__)
      case 7:
        ::transpose_impl<block_size, row_excess, 7>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
      case 6:
        ::transpose_impl<block_size, row_excess, 6>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
      case 5:
        ::transpose_impl<block_size, row_excess, 5>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
      case 4:
        ::transpose_impl<block_size, row_excess, 4>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
#endif
#if defined(__AVX__) or defined(__AVX512F__)
      case 3:
        ::transpose_impl<block_size, row_excess, 3>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
      case 2:
        ::transpose_impl<block_size, row_excess, 2>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
#endif
#if defined(__SSE2__) or defined(__AVX__)
      case 1:
        ::transpose_impl<block_size, row_excess, 1>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
#endif
      case 0:
        ::transpose_impl<block_size, row_excess, 0>(
            matrix_transpose, matrix, number_of_rows, number_of_columns);
        break;
      default:
        ERROR("Can't determine the excess number of columns.");
    };
  };
  switch (number_of_rows % static_cast<int32_t>(block_size)) {
#if defined(__AVX512F__)
    case 7:
      forward_to_impl(std::integral_constant<uint32_t, 7>{});
      break;
    case 6:
      forward_to_impl(std::integral_constant<uint32_t, 6>{});
      break;
    case 5:
      forward_to_impl(std::integral_constant<uint32_t, 5>{});
      break;
    case 4:
      forward_to_impl(std::integral_constant<uint32_t, 4>{});
      break;
#endif
#if defined(__AVX__) or defined(__AVX512F__)
    case 3:
      forward_to_impl(std::integral_constant<uint32_t, 3>{});
      break;
    case 2:
      forward_to_impl(std::integral_constant<uint32_t, 2>{});
      break;
#endif
#if defined(__SSE2__) or defined(__AVX__)
    case 1:
      forward_to_impl(std::integral_constant<uint32_t, 1>{});
      break;
#endif
    case 0:
      forward_to_impl(std::integral_constant<uint32_t, 0>{});
      break;
    default:
      ERROR("Can't determine the excess number of rows.");
  };
}
}  // namespace detail
