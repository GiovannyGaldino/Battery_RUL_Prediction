"""
describe_project_tree.py

Create a complete inventory of folders, subfolders, and files inside a project root.

This script is read-only:
- it does not delete files;
- it does not move files;
- it does not rename files;
- it only scans metadata and writes inventory reports.

Default project root:
<project-root>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_PROJECT_ROOT = "."


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".svg", ".eps", ".pdf"
}

TABLE_EXTENSIONS = {
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".feather", ".pkl", ".pickle"
}

REPORT_EXTENSIONS = {
    ".json", ".md", ".txt", ".log", ".html", ".xml", ".yaml", ".yml"
}

CODE_EXTENSIONS = {
    ".py", ".ipynb", ".bat", ".ps1", ".sh", ".sql", ".r", ".m"
}

ARCHIVE_EXTENSIONS = {
    ".zip", ".rar", ".7z", ".tar", ".gz"
}

DOCUMENT_EXTENSIONS = {
    ".docx", ".doc", ".pdf", ".pptx", ".ppt", ".dot", ".dotx"
}


@dataclass
class InventoryRecord:
    item_type: str
    name: str
    extension: str
    category: str
    relative_path: str
    absolute_path: str
    parent_relative_path: str
    depth: int
    size_bytes: int
    size_mb: float
    created_at: str
    modified_at: str
    top_level_folder: str
    is_empty_dir: Optional[bool]
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe all folders, subfolders, and files in the Batteries project."
    )

    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Root folder to scan.",
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Folder where reports will be saved. "
            "Default: <project-root>\\reports\\project_inventory"
        ),
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional maximum depth to scan. Default: scan everything.",
    )

    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden folders and files.",
    )

    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="Compute SHA256 hash for files. Slower. Disabled by default.",
    )

    parser.add_argument(
        "--max-hash-size-mb",
        type=float,
        default=100.0,
        help="Maximum file size for SHA256 hashing, in MB. Default: 100 MB.",
    )

    return parser.parse_args()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_datetime_from_timestamp(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:
        return ""


def is_hidden(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True

    if os.name == "nt":
        try:
            import ctypes

            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            if attrs == -1:
                return False
            return bool(attrs & 2)
        except Exception:
            return False

    return False


def safe_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def get_depth(path: Path, root: Path) -> int:
    rel = safe_relative_path(path, root)
    if rel == ".":
        return 0
    return len(Path(rel).parts)


def get_top_level_folder(path: Path, root: Path) -> str:
    rel = safe_relative_path(path, root)
    parts = Path(rel).parts
    if not parts:
        return "."
    return parts[0]


def classify_item(path: Path, item_type: str) -> str:
    path_text = str(path).lower()
    ext = path.suffix.lower()

    if item_type == "directory":
        if "\\raw" in path_text or "/raw" in path_text:
            return "raw_data_folder"
        if "\\interim" in path_text or "/interim" in path_text:
            return "interim_data_folder"
        if "\\processed" in path_text or "/processed" in path_text:
            return "processed_data_folder"
        if "\\reports" in path_text or "/reports" in path_text:
            return "reports_folder"
        if "\\src" in path_text or "/src" in path_text:
            return "source_code_folder"
        return "folder"

    if ext in IMAGE_EXTENSIONS:
        return "image_or_plot"

    if ext in TABLE_EXTENSIONS:
        if "\\reports" in path_text or "/reports" in path_text:
            return "tabular_report"
        return "tabular_data"

    if ext in REPORT_EXTENSIONS:
        if "\\reports" in path_text or "/reports" in path_text:
            return "report_file"
        return "text_or_metadata"

    if ext in CODE_EXTENSIONS:
        return "code"

    if ext in ARCHIVE_EXTENSIONS:
        return "archive"

    if ext in DOCUMENT_EXTENSIONS:
        return "document"

    if ext == "":
        return "no_extension"

    return "other"


def directory_size_immediate(path: Path) -> int:
    """
    Return only the immediate sum of file sizes inside a directory.
    This avoids expensive recursive size calculation for every directory.
    """
    total = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def is_empty_directory(path: Path) -> Optional[bool]:
    try:
        with os.scandir(path) as entries:
            return next(entries, None) is None
    except OSError:
        return None


def compute_sha256(path: Path, max_size_mb: float) -> str:
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return "skipped_large_file"

        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:
        return f"hash_error: {exc}"


def should_skip_path(
    path: Path,
    root: Path,
    current_output_run: Path,
    include_hidden: bool,
    max_depth: Optional[int],
) -> bool:
    if path == current_output_run or current_output_run in path.parents:
        return True

    if not include_hidden and is_hidden(path):
        return True

    if max_depth is not None:
        depth = get_depth(path, root)
        if depth > max_depth:
            return True

    return False


def make_record(path: Path, root: Path, item_type: str) -> InventoryRecord:
    error = ""
    size_bytes = 0
    created_at = ""
    modified_at = ""
    is_empty_dir_value = None

    try:
        stat = path.stat()
        created_at = iso_datetime_from_timestamp(stat.st_ctime)
        modified_at = iso_datetime_from_timestamp(stat.st_mtime)

        if item_type == "file":
            size_bytes = stat.st_size
        else:
            size_bytes = directory_size_immediate(path)
            is_empty_dir_value = is_empty_directory(path)

    except Exception as exc:
        error = str(exc)

    rel = safe_relative_path(path, root)
    parent_rel = safe_relative_path(path.parent, root)

    return InventoryRecord(
        item_type=item_type,
        name=path.name,
        extension=path.suffix.lower() if item_type == "file" else "",
        category=classify_item(path, item_type),
        relative_path=rel,
        absolute_path=str(path),
        parent_relative_path=parent_rel,
        depth=get_depth(path, root),
        size_bytes=int(size_bytes),
        size_mb=round(size_bytes / (1024 * 1024), 6),
        created_at=created_at,
        modified_at=modified_at,
        top_level_folder=get_top_level_folder(path, root),
        is_empty_dir=is_empty_dir_value,
        error=error,
    )


def scan_project(
    root: Path,
    output_run: Path,
    include_hidden: bool,
    max_depth: Optional[int],
) -> tuple[list[InventoryRecord], list[dict]]:
    records: list[InventoryRecord] = []
    errors: list[dict] = []

    # Include root folder itself.
    records.append(make_record(root, root, "directory"))

    for current_dir, dir_names, file_names in os.walk(root):
        current_path = Path(current_dir)

        # Filter directories in-place so os.walk does not enter skipped folders.
        kept_dirs = []
        for dirname in dir_names:
            dir_path = current_path / dirname

            if should_skip_path(
                dir_path,
                root=root,
                current_output_run=output_run,
                include_hidden=include_hidden,
                max_depth=max_depth,
            ):
                continue

            kept_dirs.append(dirname)

        dir_names[:] = kept_dirs

        # Register directories.
        for dirname in dir_names:
            dir_path = current_path / dirname
            try:
                records.append(make_record(dir_path, root, "directory"))
            except Exception as exc:
                errors.append(
                    {
                        "path": str(dir_path),
                        "item_type": "directory",
                        "error": str(exc),
                    }
                )

        # Register files.
        for filename in file_names:
            file_path = current_path / filename

            if should_skip_path(
                file_path,
                root=root,
                current_output_run=output_run,
                include_hidden=include_hidden,
                max_depth=max_depth,
            ):
                continue

            try:
                records.append(make_record(file_path, root, "file"))
            except Exception as exc:
                errors.append(
                    {
                        "path": str(file_path),
                        "item_type": "file",
                        "error": str(exc),
                    }
                )

    return records, errors


def save_csv(records: list[InventoryRecord], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(records[0]).keys()) if records else list(InventoryRecord.__annotations__.keys())

    with output_file.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def save_json(data: object, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_summary_tables(records: list[InventoryRecord], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_files = sum(1 for r in records if r.item_type == "file")
    total_dirs = sum(1 for r in records if r.item_type == "directory")
    total_size = sum(r.size_bytes for r in records if r.item_type == "file")

    by_extension = Counter()
    by_category = Counter()
    by_top_level = defaultdict(lambda: {"files": 0, "directories": 0, "size_bytes": 0})
    largest_files = []

    for r in records:
        by_category[r.category] += 1

        top = r.top_level_folder
        if r.item_type == "file":
            by_extension[r.extension or "[no_extension]"] += 1
            by_top_level[top]["files"] += 1
            by_top_level[top]["size_bytes"] += r.size_bytes
            largest_files.append(r)
        else:
            by_top_level[top]["directories"] += 1

    largest_files = sorted(largest_files, key=lambda x: x.size_bytes, reverse=True)[:100]

    extension_rows = [
        {"extension": ext, "count": count}
        for ext, count in by_extension.most_common()
    ]

    category_rows = [
        {"category": category, "count": count}
        for category, count in by_category.most_common()
    ]

    top_level_rows = [
        {
            "top_level_folder": folder,
            "files": values["files"],
            "directories": values["directories"],
            "size_bytes": values["size_bytes"],
            "size_mb": round(values["size_bytes"] / (1024 * 1024), 3),
            "size_gb": round(values["size_bytes"] / (1024 * 1024 * 1024), 3),
        }
        for folder, values in sorted(by_top_level.items())
    ]

    largest_rows = [
        {
            "relative_path": r.relative_path,
            "extension": r.extension,
            "category": r.category,
            "size_bytes": r.size_bytes,
            "size_mb": r.size_mb,
            "modified_at": r.modified_at,
        }
        for r in largest_files
    ]

    write_dict_rows(output_dir / "summary_by_extension.csv", extension_rows)
    write_dict_rows(output_dir / "summary_by_category.csv", category_rows)
    write_dict_rows(output_dir / "summary_by_top_level_folder.csv", top_level_rows)
    write_dict_rows(output_dir / "largest_files_top100.csv", largest_rows)

    summary = {
        "total_files": total_files,
        "total_directories": total_dirs,
        "total_items": len(records),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 3),
        "total_size_gb": round(total_size / (1024 * 1024 * 1024), 3),
        "n_extensions": len(by_extension),
        "n_categories": len(by_category),
    }

    return summary


def write_dict_rows(output_file: Path, rows: list[dict]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with output_file.open("w", encoding="utf-8-sig", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with output_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_tree_markdown(records: list[InventoryRecord], root: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sorted_records = sorted(records, key=lambda r: (r.relative_path.lower(), r.item_type))

    lines = []
    lines.append("# Project Folder Tree")
    lines.append("")
    lines.append(f"Root: `{root}`")
    lines.append("")
    lines.append("```text")
    lines.append(root.name + "/")

    for r in sorted_records:
        if r.relative_path == ".":
            continue

        indent = "  " * max(r.depth - 1, 0)
        suffix = "/" if r.item_type == "directory" else ""
        size_text = "" if r.item_type == "directory" else f"  ({r.size_mb:.3f} MB)"
        lines.append(f"{indent}├─ {r.name}{suffix}{size_text}")

    lines.append("```")
    lines.append("")

    output_file.write_text("\n".join(lines), encoding="utf-8")


def save_markdown_report(
    summary: dict,
    root: Path,
    output_run: Path,
    records: list[InventoryRecord],
    errors: list[dict],
    output_file: Path,
) -> None:
    n_empty_dirs = sum(
        1 for r in records
        if r.item_type == "directory" and r.is_empty_dir is True
    )

    n_images = sum(1 for r in records if r.category == "image_or_plot")
    n_reports = sum(1 for r in records if r.category in {"report_file", "tabular_report"})
    n_code = sum(1 for r in records if r.category == "code")
    n_archives = sum(1 for r in records if r.category == "archive")

    lines = [
        "# Project Inventory Report",
        "",
        f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"Project root: `{root}`",
        f"Output folder: `{output_run}`",
        "",
        "## General summary",
        "",
        f"- Total files: `{summary['total_files']}`",
        f"- Total directories: `{summary['total_directories']}`",
        f"- Total items: `{summary['total_items']}`",
        f"- Total file size: `{summary['total_size_gb']} GB`",
        f"- Extensions found: `{summary['n_extensions']}`",
        f"- Categories found: `{summary['n_categories']}`",
        f"- Empty directories: `{n_empty_dirs}`",
        "",
        "## Main categories",
        "",
        f"- Images / plots: `{n_images}`",
        f"- Reports / tabular reports: `{n_reports}`",
        f"- Code files: `{n_code}`",
        f"- Archive files: `{n_archives}`",
        "",
        "## Generated files",
        "",
        "- `project_inventory.csv`: complete file and folder inventory.",
        "- `project_inventory.json`: same inventory in JSON format.",
        "- `project_tree.md`: tree-like folder description.",
        "- `summary_by_extension.csv`: counts by file extension.",
        "- `summary_by_category.csv`: counts by inferred category.",
        "- `summary_by_top_level_folder.csv`: size and count by top-level folder.",
        "- `largest_files_top100.csv`: largest files found.",
        "- `scan_errors.json`: paths that could not be read, if any.",
        "",
        "## Errors",
        "",
        f"- Number of scan errors: `{len(errors)}`",
        "",
    ]

    output_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()

    if not project_root.exists():
        print(f"[ERROR] Project root does not exist: {project_root}", file=sys.stderr)
        return 1

    if args.output_root is None:
        output_root = project_root / "reports" / "project_inventory"
    else:
        output_root = Path(args.output_root).resolve()

    output_run = output_root / f"run_{now_stamp()}"
    output_run.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Output folder: {output_run}")
    print("[INFO] Scanning project tree...")

    records, errors = scan_project(
        root=project_root,
        output_run=output_run,
        include_hidden=args.include_hidden,
        max_depth=args.max_depth,
    )

    print(f"[INFO] Items found: {len(records)}")
    print("[INFO] Saving inventory files...")

    save_csv(records, output_run / "project_inventory.csv")
    save_json([asdict(r) for r in records], output_run / "project_inventory.json")
    save_json(errors, output_run / "scan_errors.json")

    summary = save_summary_tables(records, output_run)
    save_tree_markdown(records, project_root, output_run / "project_tree.md")
    save_markdown_report(
        summary=summary,
        root=project_root,
        output_run=output_run,
        records=records,
        errors=errors,
        output_file=output_run / "inventory_report.md",
    )

    if args.hash_files:
        print("[INFO] Computing file hashes...")
        hash_rows = []
        for r in records:
            if r.item_type != "file":
                continue
            hash_rows.append(
                {
                    "relative_path": r.relative_path,
                    "size_mb": r.size_mb,
                    "sha256": compute_sha256(Path(r.absolute_path), args.max_hash_size_mb),
                }
            )
        write_dict_rows(output_run / "file_hashes.csv", hash_rows)

    print("[DONE] Inventory completed.")
    print(f"[DONE] Reports saved at: {output_run}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())