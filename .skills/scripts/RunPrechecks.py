# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Deterministic CI text pre-checks for the spectre-review skill.

Rather than re-implementing SpECTRE's file checks in prose, this delegates to
the repo's own `tools/FileTestDefs.sh` `standard_checks` (long lines, tabs,
trailing whitespace, license header, `#pragma once`, `SPECTRE_TEST_CASE`,
`approx`, doxygen-in-cpp, `enable_if`, `mutable`, `Ls`, hardcoded CMake
libraries, and the rest) run over only the changed files, using the same
subset-pipe idiom as `tools/Hooks/pre-commit.sh`:

    printf '%s\\0' "${files[@]}" | run_checks "${standard_checks[@]}"

Consumes ReviewSetup.py JSON (stdin or --setup). It is SKIPPED when the CI
"Files and formatting" job is green, because that job already ran the full
`tools/CheckFiles.sh` on the whole repo. The four CI-only `ci_checks`
(iostream, TmplDebugging, TODO comments, CMakeLists add/remove) live in
CheckFiles.sh and are covered by that same CI job; pass `--full` in local mode
to additionally run `./tools/CheckFiles.sh` for them.
"""

import argparse
import json
import os
import subprocess
import sys

# Sources FileTestDefs.sh, points its grep helpers at the working tree (the
# CheckFiles.sh idiom, minus color so output stays clean for the model), then
# runs the standard checks over the file list passed as positional args.
BASH_DRIVER = r"""
top="$1"; shift
defs="$top/tools/FileTestDefs.sh"
if [ ! -f "$defs" ]; then
    echo "ERROR: $defs not found (not a SpECTRE checkout?)" >&2
    exit 2
fi
. "$defs"
staged_grep() { grep "$@"; }
pretty_grep() { grep --with-filename -n "$@"; }
printf '%s\0' "$@" | run_checks "${standard_checks[@]}"
"""


def repo_root():
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else os.getcwd()


def load_setup(path):
    if path:
        with open(path) as fh:
            return json.load(fh)
    return json.load(sys.stdin)


def check_paths(files):
    """Files to check: the head version (temp copy or working-tree path)."""
    paths = []
    for rec in files:
        if rec.get("status") == "deleted":
            continue
        head = rec.get("head_path")
        if head and os.path.exists(head):
            paths.append(head)
    return paths


def run_standard_checks(root, paths):
    proc = subprocess.run(
        ["bash", "-c", BASH_DRIVER, "bash", root] + paths,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_full_checkfiles(root):
    proc = subprocess.run(
        ["bash", os.path.join(root, "tools", "CheckFiles.sh")],
        capture_output=True,
        text=True,
        cwd=root,
    )
    return proc.returncode, proc.stdout, proc.stderr


def main():
    parser = argparse.ArgumentParser(
        description="Run SpECTRE standard_checks over changed files."
    )
    parser.add_argument(
        "--setup", default=None, help="Path to ReviewSetup JSON (else stdin)."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run tools/CheckFiles.sh (full-repo ci_checks scan).",
    )
    args = parser.parse_args()

    setup = load_setup(args.setup)
    ci = setup.get("ci", {})
    if ci.get("available") and ci.get("formatting_green"):
        print(
            "SKIPPED: CI 'Files and formatting' job is green; it already ran "
            "tools/CheckFiles.sh (standard_checks + ci_checks) on the repo."
        )
        return

    root = repo_root()
    paths = check_paths(setup.get("files", []))
    if not paths:
        print("CLEAN (no checkable files).")
        return

    rc, out, err = run_standard_checks(root, paths)
    if rc == 2:
        print(err.strip())
        sys.exit(2)
    body = out.strip()
    print(body if body else "CLEAN")
    if err.strip():
        print(err.strip(), file=sys.stderr)

    print()
    print(
        "note: iostream, TmplDebugging, TODO, and CMakeLists add/remove checks "
        "are CI-only (tools/CheckFiles.sh)."
    )
    if args.full:
        print("--- tools/CheckFiles.sh (full repo) ---")
        _, fout, ferr = run_full_checkfiles(root)
        print(fout.strip() or "CLEAN")
        if ferr.strip():
            print(ferr.strip(), file=sys.stderr)


if __name__ == "__main__":
    main()
