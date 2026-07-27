#!/usr/bin/env python3
"""Extract Java test-method source code for rows in a CSV using srcML.

The input CSV must contain these columns:
    project_url, sha_detected, java_class, test_method, module

The output preserves every input column and appends:
    source_file, test_method_code, extraction_status, extraction_error
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple
from urllib.parse import urlparse

from lxml import etree


SRC_NAMESPACE = "http://www.srcML.org/srcML/src"
NAMESPACES = {"src": SRC_NAMESPACE}
REQUIRED_COLUMNS = {
    "project_url",
    "sha_detected",
    "java_class",
    "test_method",
    "module",
}
OUTPUT_COLUMNS = [
    "source_file",
    "test_method_code",
    "extraction_status",
    "extraction_error",
]


class ExtractionError(RuntimeError):
    """An expected error that can be reported for one or more CSV rows."""


def run(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    capture_stdout: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run a command and raise an ExtractionError with useful stderr."""
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise ExtractionError(f"Could not run {command[0]!r}: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        detail = stderr or f"exit code {result.returncode}"
        raise ExtractionError(f"{' '.join(command)} failed: {detail}")
    return result


def repository_parts(project_url: str) -> Tuple[str, str]:
    """Return (owner, repository) from a GitHub-style URL."""
    cleaned = project_url.rstrip("/")
    path = urlparse(cleaned).path.strip("/")
    parts = path.split("/")
    if len(parts) < 2:
        raise ExtractionError(f"Cannot determine repository from URL: {project_url}")
    owner = parts[-2]
    repository = parts[-1]
    if repository.endswith(".git"):
        repository = repository[:-4]
    return owner, repository


def is_git_repository(path: Path) -> bool:
    return path.is_dir() and (path / ".git").exists()


def find_repository(
    projects_root: Path,
    project_url: str,
    project_name: str,
) -> Optional[Path]:
    """Find a checkout using common layouts, including the old SC layout."""
    owner, repository = repository_parts(project_url)
    candidates = [
        projects_root / repository,
        projects_root / project_name,
        projects_root / owner / repository,
        projects_root / f"{owner}_{repository}",
        projects_root / f"{owner}-{repository}",
    ]

    # Also support --projects-root pointing directly at a single repository.
    if is_git_repository(projects_root):
        candidates.insert(0, projects_root)

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if is_git_repository(resolved):
            return resolved
    return None


def git_head(repository: Path) -> str:
    result = run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
    )
    return result.stdout.decode("ascii", errors="replace").strip()


def commit_matches(head: str, requested: str) -> bool:
    requested = requested.strip()
    return bool(requested) and head.startswith(requested)


def ensure_clean_worktree(repository: Path) -> None:
    result = run(["git", "status", "--porcelain"], cwd=repository)
    if result.stdout.strip():
        raise ExtractionError(
            f"Refusing to change commits in dirty repository: {repository}"
        )


def checkout_commit(repository: Path, sha: str) -> None:
    """Check out SHA, fetching that commit from origin when necessary."""
    ensure_clean_worktree(repository)

    verify = subprocess.run(
        ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if verify.returncode != 0:
        run(
            ["git", "fetch", "--depth", "1", "origin", sha],
            cwd=repository,
            capture_stdout=False,
        )

    run(
        ["git", "checkout", "--detach", sha],
        cwd=repository,
        capture_stdout=False,
    )


def clone_repository(
    projects_root: Path,
    project_url: str,
    sha: str,
) -> Path:
    """Clone into <projects-root>/<URL repository name> and check out SHA."""
    _, repository_name = repository_parts(project_url)
    destination = (projects_root / repository_name).resolve()
    if destination.exists():
        raise ExtractionError(
            f"Cannot clone {project_url}: destination already exists: {destination}"
        )

    projects_root.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            project_url,
            str(destination),
        ],
        capture_stdout=False,
    )
    checkout_commit(destination, sha)
    return destination


def prepare_repository(
    *,
    projects_root: Path,
    project_url: str,
    project_name: str,
    sha: str,
    clone_missing: bool,
    checkout: bool,
) -> Path:
    repository = find_repository(projects_root, project_url, project_name)
    if repository is None:
        if not clone_missing:
            _, url_repository = repository_parts(project_url)
            raise ExtractionError(
                "Repository not found. Looked under "
                f"{projects_root} using {url_repository!r} and "
                f"{project_name!r}. Use --clone-missing to clone it."
            )
        repository = clone_repository(projects_root, project_url, sha)

    head = git_head(repository)
    if not commit_matches(head, sha):
        if not checkout:
            raise ExtractionError(
                f"{repository} is at {head}, but the CSV requires {sha}. "
                "Use --checkout to switch it."
            )
        checkout_commit(repository, sha)

    return repository


def normalize_module(module: str) -> Path:
    cleaned = module.strip().replace("\\", "/")
    if cleaned in {"", "."}:
        return Path()
    cleaned = cleaned.lstrip("./")
    if not cleaned or cleaned == ".":
        return Path()
    module_path = Path(cleaned)
    if module_path.is_absolute() or ".." in module_path.parts:
        raise ExtractionError(f"Unsafe module path in CSV: {module!r}")
    return module_path


def expected_source_file(
    repository: Path,
    module: str,
    java_class: str,
) -> Path:
    module_path = normalize_module(module)
    class_path = Path(*java_class.strip().split(".")).with_suffix(".java")

    # First try the standard Maven location.
    expected = repository / module_path / "src" / "test" / "java" / class_path
    if expected.is_file():
        return expected

    # Otherwise, search the entire repository by filename and package path.
    matches = []
    for candidate in repository.rglob(class_path.name):
        if not candidate.is_file():
            continue

        candidate_parts = candidate.parts
        package_parts = class_path.parts

        if candidate_parts[-len(package_parts):] == package_parts:
            matches.append(candidate)

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise ExtractionError(
            f"Java test file not found anywhere in repository: {class_path}"
        )

    relative_matches = [str(path.relative_to(repository)) for path in matches]
    raise ExtractionError(
        f"Multiple matching Java files found for {java_class}: "
        + "; ".join(relative_matches)
    )

def element_source(element: etree._Element) -> str:
    """Remove srcML tags while preserving their source-code text."""
    return "".join(element.itertext()).strip()


def nearest_class_ancestor(element: etree._Element) -> Optional[etree._Element]:
    ancestors = element.xpath("ancestor::src:class[1]", namespaces=NAMESPACES)
    return ancestors[0] if ancestors else None


def extract_test_method(
    source_file: Path,
    java_class: str,
    method_name: str,
    srcml_executable: str,
) -> str:
    """Convert one Java file to srcML and extract the requested method."""
    result = run([srcml_executable, str(source_file)])
    if not result.stdout.strip():
        raise ExtractionError(f"srcML produced no XML for {source_file}")

    try:
        root = etree.fromstring(result.stdout)
    except etree.XMLSyntaxError as exc:
        raise ExtractionError(f"Could not parse srcML output: {exc}") from exc

    class_name = java_class.rsplit(".", 1)[-1]
    classes = root.xpath(
        "//src:class[src:name=$class_name]",
        namespaces=NAMESPACES,
        class_name=class_name,
    )
    if not classes:
        raise ExtractionError(
            f"Class {class_name!r} was not found in {source_file.name}"
        )

    methods = []
    for class_element in classes:
        candidates = class_element.xpath(
            ".//src:function[src:name=$method_name]",
            namespaces=NAMESPACES,
            method_name=method_name,
        )
        methods.extend(
            method
            for method in candidates
            if nearest_class_ancestor(method) is class_element
        )

    if not methods:
        raise ExtractionError(
            f"Method {method_name!r} was not found in class {class_name!r}"
        )
    if len(methods) > 1:
        raise ExtractionError(
            f"Found {len(methods)} overloads named {method_name!r}; "
            "the CSV does not contain a signature to disambiguate them"
        )

    method_code = element_source(methods[0])
    if not method_code:
        raise ExtractionError(
            f"srcML found {class_name}.{method_name}, but its source was empty"
        )
    return method_code


def read_rows(input_csv: Path) -> Tuple[Sequence[str], list[Dict[str, str]]]:
    with input_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ExtractionError(f"CSV has no header: {input_csv}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ExtractionError(
                "CSV is missing required columns: " + ", ".join(sorted(missing))
            )
        return reader.fieldnames, list(reader)


def write_rows(
    output_csv: Path,
    input_columns: Iterable[str],
    rows: Iterable[Dict[str, str]],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = list(input_columns) + [
        column for column in OUTPUT_COLUMNS if column not in input_columns
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use srcML to append Java test-method source code to the supplied CSV."
        )
    )
    parser.add_argument("input_csv", type=Path, help="CSV containing the test rows")
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=Path("."),
        help="Directory containing project checkouts (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: <input-stem>_with_method_bodies.csv)",
    )
    parser.add_argument(
        "--srcml",
        default="srcml",
        help="srcML executable or path (default: srcml)",
    )
    parser.add_argument(
        "--clone-missing",
        action="store_true",
        help="Clone repositories that are not under --projects-root",
    )
    parser.add_argument(
        "--checkout",
        action="store_true",
        help=(
            "Check out sha_detected when an existing clean repository is at "
            "another commit"
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed row instead of recording all failures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    projects_root = args.projects_root.resolve()
    output_csv = (
        args.output.resolve()
        if args.output
        else input_csv.with_name(f"{input_csv.stem}_with_method_bodies.csv")
    )

    if shutil.which(args.srcml) is None and not Path(args.srcml).is_file():
        print(
            f"ERROR: srcML executable not found: {args.srcml!r}",
            file=sys.stderr,
        )
        return 2
    if (args.clone_missing or args.checkout) and shutil.which("git") is None:
        print("ERROR: git is required for cloning/checking out commits", file=sys.stderr)
        return 2

    try:
        input_columns, rows = read_rows(input_csv)
    except (OSError, ExtractionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    repository_cache: Dict[Tuple[str, str, str], object] = {}
    output_rows: list[Dict[str, str]] = []
    successes = 0
    failures = 0

    for index, original_row in enumerate(rows, start=2):
        row = dict(original_row)
        row.update(
            {
                "source_file": "",
                "test_method_code": "",
                "extraction_status": "",
                "extraction_error": "",
            }
        )

        project_url = row["project_url"].strip()
        project_name = row.get("project_name", "").strip()
        sha = row["sha_detected"].strip()
        cache_key = (project_url, project_name, sha)

        try:
            cached = repository_cache.get(cache_key)
            if cached is None:
                try:
                    cached = prepare_repository(
                        projects_root=projects_root,
                        project_url=project_url,
                        project_name=project_name,
                        sha=sha,
                        clone_missing=args.clone_missing,
                        checkout=args.checkout,
                    )
                except ExtractionError as exc:
                    cached = exc
                repository_cache[cache_key] = cached

            if isinstance(cached, ExtractionError):
                raise cached
            if not isinstance(cached, Path):
                raise ExtractionError("Internal repository-resolution error")

            source_file = expected_source_file(
                cached,
                row["module"],
                row["java_class"],
            )
            method_code = extract_test_method(
                source_file,
                row["java_class"].strip(),
                row["test_method"].strip(),
                args.srcml,
            )

            row["source_file"] = str(source_file.relative_to(cached))
            row["test_method_code"] = method_code
            row["extraction_status"] = "ok"
            successes += 1
            print(
                f"[{index - 1}/{len(rows)}] OK "
                f"{row['java_class']}#{row['test_method']}"
            )
        except (OSError, ExtractionError) as exc:
            failures += 1
            row["extraction_status"] = "failed"
            row["extraction_error"] = str(exc)
            print(
                f"[{index - 1}/{len(rows)}] FAILED "
                f"{row.get('node_id') or row['java_class'] + '#' + row['test_method']}: "
                f"{exc}",
                file=sys.stderr,
            )
            if args.fail_fast:
                output_rows.append(row)
                break

        output_rows.append(row)

    try:
        write_rows(output_csv, input_columns, output_rows)
    except OSError as exc:
        print(f"ERROR: Could not write {output_csv}: {exc}", file=sys.stderr)
        return 2

    print(
        f"Wrote {output_csv} ({successes} extracted, {failures} failed)."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
