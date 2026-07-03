# SpECTRE Code Review

**Arguments**: $ARGUMENTS
**Current branch**: !`git branch --show-current`

Perform a thorough code review of SpECTRE changes. Follow every step below
precisely. The full SpECTRE code rules reference (provide to review agents):
!`tail -n +5 .claude/rules/Cxx.md`

**Guiding principle**: deterministic, CI-covered checks are delegated to the
helper scripts in `.skills/scripts/` and are *skipped when the matching CI job
is already green*. Spend your own reasoning on the judgment CI cannot make:
logic and numerical bugs, misleading or copy-pasted comments, stale
documentation, dead code, and missing tests. Do not hand-run greps for things a
script already checks, and do not re-read files the setup script already
fetched.

## Step 1: Setup (run the setup helper once)

Parse `$ARGUMENTS` for a PR number and the optional words `clang-tidy` and
`coverage`, then run the setup helper — it fetches everything in one shot and
prints JSON:

- **Number present** (e.g. `1234`): PR mode.
  `python3 .skills/scripts/ReviewSetup.py <N> [--clang-tidy] [--coverage]`
- **No number**: Local mode.
  `python3 .skills/scripts/ReviewSetup.py local [--clang-tidy] [--coverage]`

Save the JSON to a file (e.g. `/tmp/spectre-review-setup.json`) and read it. It
contains: `mode`, PR `title`/`base_ref`/`head_ref`, the categorized `files`
(each with `status`, changed-line `ranges`, and `head_path`/`base_path` temp
copies), `ci` (`available`, `formatting_green`, `clang_tidy_green`), `temp_dir`,
and (local mode) `commits`. Use these values throughout; do not re-fetch the
diff or re-parse hunk ranges by hand.

Create a task list tracking the review steps below.

## Step 2: Formatting

Run: `python3 .skills/scripts/CheckFormatting.py --setup <setup.json>` and
report what it prints. It:
- Runs **clang-format** on every changed C++ file, confined to the changed
  ranges, and cancels clang-format-version-mismatch noise by subtracting any
  complaint that also appears when formatting the file's base version (so only
  PR-introduced formatting problems survive). clang-format is **not** run by CI,
  so this always runs; the local version is printed with a note that the CI
  container's version is authoritative.
- Runs **black**/**isort** on changed Python files, but **skips them when
  `formatting_green`** (CI already ran them repo-wide). It warns if the local
  black version differs from the pinned `23.3.0`.

## Step 3: CI Text Pre-Checks

Run: `python3 .skills/scripts/RunPrechecks.py --setup <setup.json>`. This
delegates to the repo's own `tools/FileTestDefs.sh` `standard_checks` (long
lines, tabs, trailing whitespace, license header, `#pragma once`,
`SPECTRE_TEST_CASE`, `approx`, doxygen-in-cpp, `enable_if`, `mutable`, `Ls`,
hardcoded CMake libraries, banned includes, `Py_RETURN_NONE`, `.ckLocal`, etc.)
over just the changed files. It **auto-skips when `formatting_green`** (CI's
"Files and formatting" job already ran the full `tools/CheckFiles.sh`). In local
mode you may add `--full` to also run `tools/CheckFiles.sh` for the CI-only
checks (iostream, TmplDebugging, TODO, CMakeLists add/remove).

Then do the checks the scripts do **not** cover (these need judgment):

### Include Order (C++ files in diff)
Verify, on the changed regions only:
1. (Tests) `"Framework/TestingFramework.hpp"` first, then blank line
2. (`.cpp` with `.hpp`) Corresponding `.hpp`, then blank line
3. STL/external `<headers>` alphabetical
4. Blank line
5. SpECTRE `"headers"` alphabetical

### LLM Comments
Identify comments that read like a coding agent's thinking-process notes.

### CMake add/remove (renames/new/deleted files only)
For files added/removed/renamed in this diff, confirm they are added to or
removed from the directory's `CMakeLists.txt` (alphabetically). (The bulk
CMakeLists check is CI-covered and skipped when green.)

### Commit Messages (local mode only)
Using `commits` from the setup JSON, check none starts (case-insensitive) with:
fixup, wip, fixme, deleteme, rebaseme, testing, rebase.

## Step 4: Code Review (2 Parallel Agents)

Launch 2 parallel agents. Give each the full SpECTRE code rules reference above,
the `files` list from the setup JSON (paths, `head_path`, changed `ranges`), and
tell them to read the `head_path` copies — not to re-fetch the PR or re-derive
ranges. Instruct each to flag only issues in lines the diff introduces.

### Agent A: Style, Patterns & Idioms
- Check the diff against every rule in **Banned Patterns** and **Style Rules**.
- Check for **Prefer-Library Patterns** (manual tensor loops that should use
  EagerMath).
- When you spot a suspicious pattern NOT in the checklist (e.g. a manual matrix
  operation, a loop reimplementing an existing utility), `grep -r` in
  `src/DataStructures/Tensor/EagerMath/`, `src/DataStructures/`,
  `src/NumericalAlgorithms/`, or `src/Utilities/` for an existing utility.
- For each finding: `file:line`, severity (`critical`/`important`/`suggestion`),
  explanation.

### Agent B: Bugs, Logic, Tests & Documentation
- Read each changed file's `head_path` in full for context.
- Look for: logic errors, off-by-one, uninitialized variables, NaN handling,
  race conditions, incorrect template instantiations, virtual inheritance issues
  (most-derived must init virtual bases).
- Check that new/changed public API in `.hpp` files has Doxygen documentation.
- Check that new source files have corresponding tests (`src/Foo/Bar.hpp` ->
  `tests/Unit/Foo/Test_Bar.cpp`).
- Only flag issues introduced by the diff. For each: `file:line`, severity,
  explanation.
- Check for problematic floating-point math:
  - Catastrophic cancellation: subtraction of nearly-equal values.
  - Division by near-zero (near boundaries, poles, special points).
  - Naive summation of many terms.
  - Unstable polynomial evaluation (monomial form) — prefer
    `evaluate_polynomial()` from `src/Utilities/Math.hpp` (Horner).
  - Numerically unstable quadratic formula.
  - Adding a tiny correction to a large value (magnitude loss).
  - Ensure `atan2` not `atan`, `log1p(x)` not `log(1+x)`, `expm1(x)` not
    `exp(x)-1`, `hypot(x,y)` not `sqrt(x*x+y*y)`.

## Step 5: clang-tidy (if requested)

If `clang-tidy` was in the arguments **and** `clang_tidy_green` is not already
true (skip when CI's Clang-tidy job passed):
1. Check for `build/compile_commands.json`. If missing, report that clang-tidy
   requires a configured build directory and skip.
2. For each changed `.cpp` file: `clang-tidy -p build/ FILEPATH 2>&1`.
3. Filter output to only warnings on lines in the diff (`ranges`).

## Step 6: Code Coverage (if requested)

If `coverage` was in the arguments, read `references/coverage-steps.md` and
follow those instructions exactly.

## Step 7: Self-Review Prune

Combine all findings from steps 2-6. REMOVE only clear non-issues:
- False positives (a pattern match that isn't the actual flagged issue).
- Pre-existing issues not introduced by this diff, except missing includes.
- Issues suppressed by `// NOLINT(...)` comments.
- Exact duplicates between agents or between agents and the script output.
- **Any clang-format/black/isort finding outside the changed `ranges`.**
- **Downgrade** (important -> suggestion) any formatting finding when the local
  tool version differs from the pinned/container version, per the version notes
  the scripts print.

Assign each remaining finding a confidence score (0-100). Remove findings below
50. Keep all scoring 50 or above — err toward including borderline issues.

## Step 8: Lightweight Model Critique

Spawn an agent using the cheapest available model (Claude Code: `haiku`; Codex:
`gpt-5.4-mini`). Provide it with the SpECTRE code rules reference, the pruned
findings (with scores), and a summary of what the diff does. Ask it to:
1. Score each finding 0-100 for "is this a real, actionable issue?"
2. Flag remaining false positives with reasoning.
3. Note important issues that seem to be missing.

After its feedback: remove findings scored < 40; downgrade severity for 40-60;
consider adding issues it suggests (verify first).

## Step 9: Final Report

Read `references/report-template.md` and present the report in that format.
