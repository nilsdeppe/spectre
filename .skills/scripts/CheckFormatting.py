# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Formatting checks for the spectre-review skill (changed lines only).

Consumes the JSON emitted by ReviewSetup.py (on stdin or via --setup) and
reports only real, changed-line formatting problems, avoiding the two token
sinks that plague ad-hoc runs:

  - C++ (clang-format): each changed file is formatted with `--lines=A:B`
    confined to the diff's changed ranges and `--assume-filename=<repo path>`
    so the repo `.clang-format` is used even for temp copies. Because only
    changed lines are reformatted, version-mismatch noise on untouched lines
    never appears. clang-format is NOT run by SpECTRE CI, so it always runs
    here; the local version is printed with a note that the CI container's
    version is authoritative.

  - Python (black/isort): skipped entirely when the CI "Files and formatting"
    job is green (it runs black/isort on the whole repo). Otherwise black and
    isort run with the repo config; the local black version is compared to the
    pinned 23.3.0 and a warning is emitted on mismatch.

Output is compact: a per-file diff or `CLEAN`, and `UNAVAILABLE: <tool>` if a
formatter is missing (never a crash).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

PINNED_BLACK = "23.3.0"
CPP_SUFFIXES = (".cpp", ".hpp", ".tpp")


def run(cmd, stdin_text=None):
    """Run a command; return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, input=stdin_text, capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr


def tool_version(cmd):
    rc, out, err = run(cmd)
    if rc != 0:
        return None
    text = (out or err).strip()
    m = re.search(r"(\d+\.\d+\.\d+)", text)
    return m.group(1) if m else text


def repo_root():
    rc, out, _ = run(["git", "rev-parse", "--show-toplevel"])
    return out.strip() if rc == 0 else os.getcwd()


def load_setup(path):
    if path:
        with open(path) as fh:
            return json.load(fh)
    return json.load(sys.stdin)


def read_source(rec):
    """Return the head-version text for a file record, or None."""
    head = rec.get("head_path")
    if head and os.path.exists(head):
        with open(head) as fh:
            return fh.read()
    return None


def clang_format(source, assumed, ranges=None):
    """Return clang-format's output for `source`, or None on error."""
    cmd = ["clang-format", "-style=file", f"--assume-filename={assumed}"]
    for lo, hi in ranges or []:
        cmd.append(f"--lines={lo}:{hi}")
    rc, out, _ = run(cmd, stdin_text=source)
    return out if rc == 0 else None


def complaint_hunks(source, assumed):
    """Whole-file clang-format complaints as a set of normalized hunks.

    Each hunk is reduced to the tuple of its '-'/'+' lines (line numbers and
    context dropped) so the same reformatting produces the same key regardless
    of where it sits in the file. Used to subtract version-mismatch noise that
    is already present in the base version of a file.
    """
    formatted = clang_format(source, assumed)
    if formatted is None or formatted == source:
        return set()
    return {h for h in _hunk_keys(source, formatted)}


def _hunk_keys(before, after):
    import difflib

    keys = []
    diff = difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm="", n=0
    )
    current = []
    for line in diff:
        if line.startswith("@@"):
            if current:
                keys.append(tuple(current))
            current = []
        elif line.startswith("+++") or line.startswith("---"):
            continue
        elif line and line[0] in "+-":
            current.append(line)
    if current:
        keys.append(tuple(current))
    return keys


def check_cpp(files, root):
    """clang-format on changed lines, cancelling base version-mismatch noise.

    A file is reported only for hunks that are NOT already produced by
    formatting its base version (those are clang-format-version disagreements,
    not PR problems). Findings are confined to the diff's changed ranges via
    `--lines`.
    """
    if shutil.which("clang-format") is None:
        return ["UNAVAILABLE: clang-format"]
    reports = []
    for rec in files:
        if rec["category"] != "cpp" or rec["status"] == "deleted":
            continue
        source = read_source(rec)
        ranges = rec.get("ranges", [])
        if source is None or not ranges:
            continue
        assumed = os.path.join(root, rec["path"])
        formatted = clang_format(source, assumed, ranges)
        if formatted is None:
            reports.append(f"{rec['path']}: clang-format error")
            continue
        if formatted == source:
            continue
        # Subtract noise: hunks that also arise from the base version.
        base_noise = set()
        base_path = rec.get("base_path")
        if base_path and os.path.exists(base_path):
            with open(base_path) as fh:
                base_noise = complaint_hunks(fh.read(), assumed)
        real = [
            h
            for h in _hunk_keys(source, formatted)
            if tuple(h) not in base_noise
        ]
        if real:
            reports.append(f"{rec['path']}:\n" + render_hunks(real))
    return reports


def render_hunks(hunks, max_lines=40):
    out = []
    for h in hunks:
        out.extend(h)
    if len(out) > max_lines:
        out = out[:max_lines] + [f"... (+{len(out) - max_lines} more lines)"]
    return "\n".join(out)


def check_python(files, root, config_dir):
    """Run black/isort with repo config on changed Python files."""
    py = [
        rec
        for rec in files
        if rec["category"] == "py"
        and rec["status"] != "deleted"
        and "/external/" not in ("/" + rec["path"])
        and rec.get("head_path")
    ]
    if not py:
        return []
    reports = []
    if shutil.which("black") is None:
        reports.append("UNAVAILABLE: black")
    else:
        local = tool_version(["black", "--version"])
        if local and local != PINNED_BLACK:
            reports.append(
                f"NOTE: local black {local} != pinned {PINNED_BLACK}; "
                "CI is authoritative for any finding below."
            )
        pyproject = os.path.join(config_dir, "pyproject.toml")
        for rec in py:
            cmd = ["black", "--check", "--diff"]
            if os.path.exists(pyproject):
                cmd += ["--config", pyproject]
            cmd.append(rec["head_path"])
            rc, out, err = run(cmd)
            if rc != 0 and (out.strip() or "would reformat" in err):
                reports.append(f"{rec['path']} (black):\n{out.strip()}")
    if shutil.which("isort") is None:
        reports.append("UNAVAILABLE: isort")
    else:
        for rec in py:
            cmd = [
                "isort",
                "--check-only",
                "--diff",
                "--settings-path",
                config_dir,
                rec["head_path"],
            ]
            rc, out, err = run(cmd)
            if rc != 0 and out.strip():
                reports.append(f"{rec['path']} (isort):\n{out.strip()}")
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Changed-line formatting checks for spectre-review."
    )
    parser.add_argument(
        "--setup", default=None, help="Path to ReviewSetup JSON (else stdin)."
    )
    args = parser.parse_args()

    setup = load_setup(args.setup)
    files = setup.get("files", [])
    root = repo_root()
    ci = setup.get("ci", {})
    formatting_green = ci.get("available") and ci.get("formatting_green")

    print("=== C++ (clang-format, changed lines only) ===")
    version = tool_version(["clang-format", "--version"])
    print(f"local clang-format: {version or 'not found'}")
    print(
        "note: the sxscollaboration/spectre:dev container version is "
        "authoritative; verify residual diffs with the pre-commit hook."
    )
    cpp_reports = check_cpp(files, root)
    print("\n".join(cpp_reports) if cpp_reports else "CLEAN")

    print()
    print("=== Python (black + isort) ===")
    if formatting_green:
        print("SKIPPED: CI 'Files and formatting' job is green.")
    else:
        py_reports = check_python(files, root, root)
        print("\n".join(py_reports) if py_reports else "CLEAN")


if __name__ == "__main__":
    main()
