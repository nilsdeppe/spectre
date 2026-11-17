\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# SpECTRE Code Review: Powell and GLM terms

**Reviewing**: local changes on branch `div_cleaning_claude_refactor`, commits
`e7dcb24320..0c4c6dbb67` (8 commits)

**Files changed**: 93 (78 C++, 10 Python, 4 CMake, 1 bib)

Nothing was built or run; all findings come from reading the code. The critical
item was confirmed directly against the source.

Paths below abbreviate `VDC` = `src/Evolution/Systems/GrMhd/ValenciaDivClean`
and `GhVDC` = `src/Evolution/Systems/GrMhd/GhValenciaDivClean`.

## Formatting

- clang-format wants a different wrap at
  `VDC/FiniteDifference/ReconstructWork.tpp:106-107` (`make_not_null(` on its
  own line, then `&get<...LapseTimesbOverW>(*vars_on_face)),`).
- black and isort are clean.

## Critical Issues

1. **DG boundary terms for the Powell/GLM terms are inconsistent**
   `VDC/BoundaryCorrections/NonConservativeTerms.cpp:110-130`
   - The exterior package data uses the neighbour's outward normal, so
     `n_ext = -n_int` (the header says this itself). Substituting that into the
     code:
     - **`TildePhi`, `TildeB`, `TildeS`:** the terms become
       `1/4 * jump(field) * n.(coef_int - coef_ext)`, a product of two jumps.
       They vanish whenever `alpha v^i` or `alpha b_i/W` is continuous, so the
       interface coupling for the volume terms is effectively absent.
     - **`TildeTau`, last term:** it becomes
       `1/4 (Btilde_int + Btilde_ext).n (a_int + a_ext)` with
       `a = alpha B^m v_m`. This is non-zero for identical states on both
       sides, and it is added in both strong and weak form.
   - A uniform magnetised moving fluid with `B.v != 0` therefore gets a
     non-zero `dt TildeTau` at every DG face.
   - The header note at `NonConservativeTerms.hpp:111-118` presents this as
     intended ("cancels the magnetic field term of the flux"), but a
     strong-form correction must vanish for continuous states.
   - `Test_Hll.cpp:73`, `Test_Rusanov.cpp:72` and
     `Test_ProductOfCorrections.cpp:272` were switched to
     `ZeroOnSmoothSolution::No`, which turns off the check that would catch
     this. The only remaining coverage is `NonConservativeTerms.py`, which
     mirrors the C++ line by line.
   - Suggested fix: use the path-conservative form with everything expressed
     through `n_int`:
     - `G(TildePhi) += 1/2 (alpha v.n)_int * jump(TildePhi)`
     - `G(TildeB^i) += 1/2 (alpha v^i)_int * jump(TildeB).n`
     - `G(TildeS_i) += 1/2 (alpha b_i/W)_int * jump(TildeB).n`
     - `G(TildeTau) += 1/2 (alpha B.v)_int * jump(TildeB).n
       + 1/2 (TildeB.n)_int * jump(TildePhi)`

     Or keep averaged coefficients and flip the sign of each exterior
     normal-contracted factor. Then restore `ZeroOnSmoothSolution::Yes`, exempt
     these terms from the weak-form antisymmetry check (they are legitimately
     not antisymmetric), and add a test that a constant magnetised state gives
     zero correction.

## Important Issues

1. **`ResetInitialDataIfOnDg` is untested, misnamed and duplicated**
   `src/Evolution/DgSubcell/Actions/Initialize.hpp:693-751`
   - Nothing under `tests/` references it, including the numeric-initial-data
     branch.
   - It re-evaluates on whichever grid is active, not only on DG.
   - The `apply` body and tag lists are copied from `SetSubcellGrid` (lines
     110-140, 263-276).
   - The docs say it sets the evolved variables, but for systems with
     primitives it only sets primitives; it is correct only because
     `UpdateConservatives` follows in the action list, which is not documented.
   - Suggested fix: factor out the shared dispatch, rename (e.g.
     `SetInitialDataOnActiveGrid`), document the ordering requirement, add a
     test.

2. **No direct test for the boundary non-conservative terms**
   - `VDC/BoundaryCorrections/NonConservativeTerms.hpp` has only a Python
     mirror, reached through `Test_Hll` / `Test_Rusanov`. There is no
     `Test_NonConservativeTerms.cpp` in the test `BoundaryCorrections/`
     directory.
   - More broadly, no test checks the Powell/GLM coefficients independently:
     C++ and Python use the same expressions everywhere, and
     `Test_SmoothFlow.cpp` only verifies amplitude 0.

3. **21 `DataVector` allocations per FD time-derivative call**
   `VDC/Subcell/TimeDerivative.hpp:107-109`,
   `GhVDC/Subcell/TimeDerivative.hpp:641-643`
   - `face_tilde_b`, `face_tilde_phi` and `face_conormal` are three
     `std::array`s of tensors. The project rules ask for a `Variables` here.

## Suggestions

1. **GLM term in the `TildeTau` equation needs a derivation check**
   `VDC/TimeDerivativeTerms.cpp:175`,
   `VDC/Subcell/NonConservativeTerms.cpp:~103`
   - `tilde_b^i * d_i tilde_phi` is a product of two densitised quantities with
     no lapse, unlike every other right-hand-side term. It may need to be
     `alpha B^i` with a matching gradient; this was not derived during the
     review. Tests cannot detect this either way.

2. **FD Powell/GLM terms are always second order**
   `VDC/Subcell/NonConservativeTerms.cpp:71-89`
   - `Subcell/Test_TimeDerivative.cpp:589-595` now restricts the "error
     decreases with FD order" check to `TildeD` / `TildeYe` to accommodate
     this. Either use the FD derivative order or state the limitation in the
     doxygen.

3. **`std::function` in the FD hot path**
   `VDC/Subcell/NonConservativeTerms.hpp:103`
   - The Cartesian/Cartoon lambda is also duplicated between
     `VDC/Subcell/TimeDerivative.hpp:644-665` and
     `GhVDC/Subcell/TimeDerivative.hpp:471-490`. Passing `div_tilde_b` in, or
     templating on the callable, removes both.

4. **Duplicated blocks between the two `Subcell/TimeDerivative.hpp` files**
   - Face averaging (VDC 286-299, GhVDC 715-727), conormal rescaling (389-392,
     815-818), and zeroing of the package fields. Small shared helpers would
     cover them.

5. **Recomputed quantities** `VDC/Subcell/NonConservativeTerms.cpp:85-96`
   - `v_i`, `B.v` and `B_i` are recomputed although the GhValenciaDivClean
     caller already has them.

6. **Manual tensor loops where a utility exists**
   - `VDC/LapseTimesbOverW.cpp:27-33`: `raise_or_lower_index`.
   - `VDC/BoundaryCorrections/NonConservativeTerms.cpp:101-108` and
     `VDC/Subcell/NonConservativeTerms.cpp:70-73`: `dot_product` or
     `tenex::evaluate`.

7. **Divergence accumulated per diagonal component**
   `VDC/TimeDerivativeTerms.cpp:161-172`
   - Each coefficient product is evaluated three times; a `DivTildeB` temporary
     would match the subcell version.

8. **Thin test**
   `Subcell/Test_NonConservativeTerms.cpp:29` (under
   `tests/Unit/Evolution/Systems/GrMhd/ValenciaDivClean/`)
   - Only `comp_dim == 3` with a constant diagonal Jacobian is covered; the
     Cartoon branch is not. The body sits inline in the `SPECTRE_TEST_CASE`
     instead of an anonymous-namespace helper.

9. **Inaccurate comment** `GhVDC/TimeDerivativeTerms.hpp:152-160`
   - Zeroing the two temporaries does not disable this element's boundary
     contribution; `alpha v^i` and `n_i` are still packaged.

10. **SmoothFlow monopole option**
    `src/PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp:55-64`
    - `MagneticMonopoleAmplitude` is a new required option. No in-repo input
      file uses GRMHD SmoothFlow, but external ones will break.
    - With a non-zero amplitude the class is still a `MarkAsAnalyticSolution`,
      and `B^z = A z exp(-z^2/2)` is not periodic although the class is
      described as periodic.

11. **Velocity norm observable**
    `EvolveValenciaDivClean.hpp:256` (under
    `src/Evolution/Executables/GrMhd/ValenciaDivClean/`)
    - `PointwiseL2Norm(SpatialVelocity)` is the Euclidean norm of `v^i`, not
      the metric norm; worth a note for curved backgrounds.

## CI Pre-Check Warnings

None. No long lines, tabs, trailing whitespace, banned patterns, TODOs, missing
license headers or `#pragma once`; the four changed CMake lists stay
alphabetical; commit messages are fine.

## Checked and found consistent

The variable reordering: `System`, fluxes, `ConservativeFromPrimitive`, all
`dg_ghost` signatures, `SpectralFilter.cpp`, reconstruction tag lists,
package-data order in Hll/Rusanov and `ProductOfCorrections` all match the new
order, and the consumers outside the diff are tag-based. Checkpoint or
volume-data compatibility across the reorder was not checked.
