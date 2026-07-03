# Distributed under the MIT License.
# See LICENSE.txt for details.

"""One-shot setup for the spectre-review skill.

Acquires everything the review needs in a single invocation and prints a
compact JSON object to stdout so the model reads it once instead of issuing
many exploratory tool calls:

  - PR or local mode detection and metadata (title, base/head refs).
  - The list of changed files, categorized (cpp/c/py/fortran/perl/yaml/h5/
    cmake/other).
  - Per-file changed-line ranges (parsed from the diff, expanded by +/-4).
  - For PR mode, the PR-head version of each changed file dumped under a temp
    directory that mirrors the repo layout (so `--assume-filename` style-lookup
    and dirname-based checks resolve correctly). Local mode uses the working
    tree directly.
  - CI job states via `gh pr checks`, exposing `formatting_green`
    (the "Files and formatting" job) and `clang_tidy_green`. These let the
    caller skip deterministic checks that a green CI job already verified.

The script never mutates the working tree or checks out the PR branch; it
fetches the PR head into a private ref and reads blobs out of it with
`git show`.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

DEFAULT_REPO = "sxs-collaboration/spectre"
CONTEXT_LINES = 4

CPP_SUFFIXES = (".cpp", ".hpp", ".tpp")
C_SUFFIXES = (".c", ".h")
FORTRAN_SUFFIXES = (".f", ".F", ".f77", ".f90", ".F90", ".for", ".FOR")
PERL_SUFFIXES = (".pl", ".pm", ".perl")
YAML_SUFFIXES = (".yaml", ".yml")
H5_SUFFIXES = (".h5", ".hdf5")
CMAKE_SUFFIXES = (".cmake",)


def run(cmd, check=False):
    """Run a command, returning (returncode, stdout, stderr) as text."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(
            f"Error running {' '.join(cmd)}: {result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)
    return result.returncode, result.stdout, result.stderr


def git(args, check=False):
    """Run a git command and return stdout (empty string on failure)."""
    rc, out, _ = run(["git"] + args, check=check)
    return out if rc == 0 else ""


def repo_root():
    root = git(["rev-parse", "--show-toplevel"], check=True).strip()
    return root


def categorize(path):
    if path.endswith(CPP_SUFFIXES):
        return "cpp"
    if path.endswith(C_SUFFIXES):
        return "c"
    if path.endswith(".py"):
        return "py"
    if path.endswith(FORTRAN_SUFFIXES):
        return "fortran"
    if path.endswith(PERL_SUFFIXES):
        return "perl"
    if path.endswith(YAML_SUFFIXES):
        return "yaml"
    if path.endswith(H5_SUFFIXES):
        return "h5"
    if (
        os.path.basename(path) == "CMakeLists.txt"
        or path.endswith(CMAKE_SUFFIXES)
    ):
        return "cmake"
    return "other"


def parse_diff_ranges(diff_text):
    """Map each file to a list of [start, end] changed-line ranges.

    Ranges are taken from the '+' side of each hunk header and expanded by
    +/-CONTEXT_LINES lines (clamped at 1). Overlapping ranges are merged.
    """
    ranges = {}
    current = None
    file_re = re.compile(r"^\+\+\+ b/(.*)$")
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    for line in diff_text.splitlines():
        m = file_re.match(line)
        if m:
            current = m.group(1)
            ranges.setdefault(current, [])
            continue
        m = hunk_re.match(line)
        if m and current is not None:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            if count == 0:
                # Pure deletion hunk; anchor a single line for context.
                count = 1
            lo = max(1, start - CONTEXT_LINES)
            hi = start + count - 1 + CONTEXT_LINES
            ranges[current].append([lo, hi])
    return {f: merge_ranges(rs) for f, rs in ranges.items()}


def statuses_from_diff(diff_text):
    """Map each file in a unified diff to added/modified/deleted.

    Uses the per-file 'diff --git a/x b/y' header plus the 'new file' /
    'deleted file' markers that follow it.
    """
    statuses = {}
    current = None
    header_re = re.compile(r"^diff --git a/.* b/(.*)$")
    for line in diff_text.splitlines():
        m = header_re.match(line)
        if m:
            current = m.group(1)
            statuses[current] = "modified"
        elif current is not None:
            if line.startswith("new file mode"):
                statuses[current] = "added"
            elif line.startswith("deleted file mode"):
                statuses[current] = "deleted"
    return statuses


def merge_ranges(rs):
    if not rs:
        return []
    rs = sorted(rs)
    merged = [rs[0][:]]
    for lo, hi in rs[1:]:
        if lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def find_pr_remote(repo):
    """Return the git remote that hosts the given owner/repo.

    Prefers a remote whose URL matches `repo`, then falls back to `upstream`
    then `origin` (GitHub stores PR heads under the base repo's remote).
    """
    _, out, _ = run(["git", "remote", "-v"])
    remotes = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            remotes[parts[0]] = parts[1]
    owner_name = repo.lower()
    for name, url in remotes.items():
        if owner_name in url.lower():
            return name
    for name in ("upstream", "origin"):
        if name in remotes:
            return name
    return None


def fetch_pr_head(pr_number, repo):
    """Fetch the PR head into a private ref; return the ref name or None."""
    remote = find_pr_remote(repo)
    if remote is None:
        return None
    ref = f"refs/spectre-review/{pr_number}"
    rc, _, err = run(
        [
            "git",
            "fetch",
            "--quiet",
            remote,
            f"+refs/pull/{pr_number}/head:{ref}",
        ]
    )
    if rc != 0:
        print(f"Warning: git fetch failed: {err.strip()}", file=sys.stderr)
        return None
    return ref


def merge_base_sha(head_ref, base_ref, repo):
    """Return the merge-base of the PR head and its base branch, or None.

    Fetches the base branch from the PR's remote so the comparison works even
    when the local checkout is on an unrelated branch.
    """
    remote = find_pr_remote(repo)
    if remote is None or not base_ref:
        return None
    rc, _, _ = run(["git", "fetch", "--quiet", remote, base_ref])
    if rc != 0:
        return None
    rc, out, _ = run(["git", "merge-base", head_ref, "FETCH_HEAD"])
    return out.strip() if rc == 0 else None


def dump_blobs(ref, files, dest_root):
    """Write the `ref` version of each file under dest_root/<path>.

    Returns a dict path -> absolute temp path. Files with no blob at `ref`
    (e.g. deleted, renamed away, or newly added) are skipped silently.
    """
    out = {}
    for path in files:
        rc, content, _ = run(["git", "show", f"{ref}:{path}"])
        if rc != 0:
            continue
        dest = os.path.join(dest_root, path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as fh:
            fh.write(content)
        out[path] = dest
    return out


def gh_json(cmd):
    rc, out, _ = run(cmd)
    if rc != 0 or not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def collect_ci(pr_number, repo):
    """Return CI summary or {'available': False} if gh is unavailable."""
    data = gh_json(
        [
            "gh",
            "pr",
            "checks",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "name,state",
        ]
    )
    if data is None:
        return {"available": False}
    jobs = {j["name"]: j["state"] for j in data}
    fmt = jobs.get("Files and formatting")
    clang_tidy = [s for n, s in jobs.items() if n.startswith("Clang-tidy")]
    return {
        "available": True,
        "formatting_green": fmt == "SUCCESS",
        "clang_tidy_green": bool(clang_tidy) and all(
            s == "SUCCESS" for s in clang_tidy
        ),
        "jobs": jobs,
    }


def build_files(paths_statuses, ranges, head_paths, base_paths, root):
    """Assemble the per-file records for the JSON output."""
    files = []
    for path, status in sorted(paths_statuses.items()):
        rec = {
            "path": path,
            "category": categorize(path),
            "status": status,
            "ranges": ranges.get(path, []),
        }
        if path in head_paths:
            rec["head_path"] = head_paths[path]
        elif status != "deleted":
            # Local mode: the working-tree file is the head version.
            abs_path = os.path.join(root, path)
            if os.path.exists(abs_path):
                rec["head_path"] = abs_path
        if path in base_paths:
            rec["base_path"] = base_paths[path]
        files.append(rec)
    return files


def name_status(diff_range):
    """Return {path: status} from `git diff --name-status <range>`."""
    out = git(["diff", "--name-status"] + diff_range)
    result = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code = parts[0]
        # Renames/copies look like R100\told\tnew; take the new path.
        path = parts[-1]
        status = {
            "A": "added",
            "M": "modified",
            "D": "deleted",
        }.get(code[0], "modified")
        result[path] = status
    return result


def setup_pr(pr_number, repo, flags):
    rc, out, _ = run(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "title,body,headRefName,baseRefName",
        ]
    )
    meta = {}
    if rc == 0 and out.strip():
        try:
            meta = json.loads(out)
        except json.JSONDecodeError:
            meta = {}

    _, diff_text, _ = run(["gh", "pr", "diff", str(pr_number), "--repo", repo])
    ranges = parse_diff_ranges(diff_text)
    paths_statuses = statuses_from_diff(diff_text)

    ref = fetch_pr_head(pr_number, repo)
    root = repo_root()
    temp_dir = tempfile.mkdtemp(prefix=f"spectre-review-{pr_number}-")
    head_paths = {}
    base_paths = {}
    if ref is not None:
        head_root = os.path.join(temp_dir, "head")
        head_paths = dump_blobs(ref, paths_statuses.keys(), head_root)
        base_sha = merge_base_sha(ref, meta.get("baseRefName", ""), repo)
        if base_sha is not None:
            base_root = os.path.join(temp_dir, "base")
            base_paths = dump_blobs(base_sha, paths_statuses.keys(), base_root)

    files = build_files(paths_statuses, ranges, head_paths, base_paths, root)
    return {
        "mode": "pr",
        "pr_number": pr_number,
        "repo": repo,
        "title": meta.get("title", ""),
        "base_ref": meta.get("baseRefName", ""),
        "head_ref": meta.get("headRefName", ""),
        "head_fetched": ref is not None,
        "flags": flags,
        "ci": collect_ci(pr_number, repo),
        "temp_dir": temp_dir,
        "files": files,
    }


def setup_local(base, flags):
    root = repo_root()
    # Committed changes vs base, plus staged and unstaged working-tree edits.
    committed = name_status([f"{base}...HEAD"])
    staged = name_status(["--cached"])
    unstaged = name_status([])
    if not (committed or staged or unstaged):
        # Fall back to the previous commit if nothing else is pending.
        committed = name_status(["HEAD~1...HEAD"])
        base = "HEAD~1"

    paths_statuses = {}
    for src in (committed, staged, unstaged):
        paths_statuses.update(src)

    diff_text = (
        git(["diff", f"{base}...HEAD"])
        + "\n"
        + git(["diff", "--cached"])
        + "\n"
        + git(["diff"])
    )
    ranges = parse_diff_ranges(diff_text)

    # Dump the base-branch version of each file for diff-of-diffs noise
    # cancellation in CheckFormatting. head is the working tree (real paths).
    temp_dir = tempfile.mkdtemp(prefix="spectre-review-local-")
    base_paths = dump_blobs(
        base, paths_statuses.keys(), os.path.join(temp_dir, "base")
    )
    files = build_files(paths_statuses, ranges, {}, base_paths, root)
    commits = [
        s for s in git(["log", "--format=%s", f"{base}..HEAD"]).splitlines()
    ]
    return {
        "mode": "local",
        "base_ref": base,
        "flags": flags,
        "ci": {"available": False},
        "temp_dir": temp_dir,
        "files": files,
        "commits": commits,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Set up a spectre-review run and emit JSON."
    )
    parser.add_argument(
        "target",
        help="PR number, or 'local' for the working-tree/committed changes.",
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help="owner/repo for PR mode."
    )
    parser.add_argument(
        "--base", default="develop", help="Base branch for local mode."
    )
    parser.add_argument(
        "--clang-tidy", action="store_true", help="Caller requested clang-tidy."
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Caller requested coverage."
    )
    args = parser.parse_args()

    flags = {"clang_tidy": args.clang_tidy, "coverage": args.coverage}
    if args.target.isdigit():
        result = setup_pr(int(args.target), args.repo, flags)
    else:
        result = setup_local(args.base, flags)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
