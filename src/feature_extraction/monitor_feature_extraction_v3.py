"""
monitor_feature_extraction_v3.py

Monitor script for build_hierarchical_features_v3.py.

Purpose
-------
Run this script in a second PowerShell terminal while the feature extraction is
running. It does not modify the extraction outputs.

It reports:
- overall progress in files and input rows
- completed, running, pending and failed files
- dataset/base, phase folder, input file and output file
- current battery and cycle when this can be inferred from temporary output files
  or progress_logs
- list of completed files
- CSV snapshots for later inspection

It works even if the extractor was not patched for verbose progress. In that case,
current battery/cycle is inferred from the temporary Parquet file when readable.
If no temp file is readable, the script still reports completed/pending/running
by file.

Typical PowerShell use
----------------------

$args_monitor = @(
  ".\\src\\feature_extraction\\monitor_feature_extraction_v3.py",
  "--input-root", "C:\\Artigo Baterias de Litio\\Batteries\\data\\interim",
  "--output-root", "C:\\Artigo Baterias de Litio\\Batteries\\data\\processed\\features_hierarchical_v3",
  "--reports-root", "C:\\Artigo Baterias de Litio\\Batteries\\reports\\feature_extraction_v3",
  "--refresh-seconds", "30",
  "--watch"
)

& C:\\Users\\Giovanny\\anaconda3\\envs\\batteries\\python.exe @args_monitor

For one snapshot only, remove --watch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import fastparquet
except Exception:
    fastparquet = None


DEFAULT_INPUT_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/interim")
DEFAULT_OUTPUT_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/data/processed/features_hierarchical_v3")
DEFAULT_REPORTS_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/reports/feature_extraction_v3")
DEFAULT_MONITOR_ROOT = Path(r"C:/Artigo Baterias de Litio/Batteries/reports/feature_extraction_v3_monitor")

PHASE_FOLDERS = ("charge", "discharge")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_id() -> str:
    return datetime.now().strftime("monitor_%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def short_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def safe_filename(value: str, max_len: int = 140) -> str:
    value = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value)).strip("_")
    return (value or "file")[:max_len]


def infer_dataset_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"_(charge|discharge).*", "", stem)
    return stem.upper() if stem else "UNKNOWN"


def compact(value: Any, max_len: int = 42) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parquet_row_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None

    if fastparquet is not None:
        try:
            pf = fastparquet.ParquetFile(str(path))
            return int(sum(rg.num_rows for rg in pf.fmd.row_groups))
        except Exception:
            pass

    try:
        return int(len(pd.read_parquet(path, engine="fastparquet", columns=[])))
    except Exception:
        pass

    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def parquet_columns(path: Path) -> List[str]:
    if fastparquet is not None:
        try:
            return list(fastparquet.ParquetFile(str(path)).columns)
        except Exception:
            return []
    return []


def latest_run_dir(reports_root: Path) -> Optional[Path]:
    if not reports_root.exists():
        return None

    candidates = [p for p in reports_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        return None

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def parse_dataset_args(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part.upper())
    return list(dict.fromkeys(out))


def discover_jobs(
    input_root: Path,
    output_root: Path,
    phase_dirs: Sequence[str],
    datasets: Sequence[str],
) -> pd.DataFrame:
    wanted = {d.upper() for d in datasets if d}
    records: List[Dict[str, Any]] = []

    for phase_folder in phase_dirs:
        folder = input_root / phase_folder
        if not folder.exists():
            continue

        for input_path in sorted(folder.glob("*.parquet"), key=lambda p: str(p).lower()):
            dataset = infer_dataset_from_filename(input_path)

            if wanted and dataset.upper() not in wanted:
                continue

            job_id = short_hash(f"{phase_folder}/{input_path.name}")
            out_dir = output_root / phase_folder / dataset.lower()
            output_path = out_dir / f"{input_path.stem}__{job_id}.features.parquet"
            done_path = Path(str(output_path) + ".done")
            error_path = out_dir / f"{input_path.stem}__{job_id}.error.txt"

            records.append(
                {
                    "job_id": job_id,
                    "dataset": dataset,
                    "phase_folder": phase_folder,
                    "input_file": str(input_path),
                    "input_name": input_path.name,
                    "output_file": str(output_path),
                    "done_file": str(done_path),
                    "error_file": str(error_path),
                    "expected_input_rows": parquet_row_count(input_path),
                }
            )

    return pd.DataFrame(records)


def read_latest_status_events(run_dir: Optional[Path]) -> pd.DataFrame:
    if run_dir is None:
        return pd.DataFrame()

    status_path = run_dir / "status.jsonl"
    if not status_path.exists():
        return pd.DataFrame()

    records = []
    try:
        with status_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame(records)


def find_tmp_files(output_path: Path) -> List[Path]:
    if not output_path.parent.exists():
        return []

    pattern = f".{output_path.name}.tmp.*"
    return sorted(output_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def read_tmp_progress(tmp_path: Path) -> Dict[str, Any]:
    """
    Try to read a temporary output Parquet file created during extraction.

    This is best-effort. If the file is being written exactly at this moment,
    fastparquet may fail. That is not a problem; the next monitor refresh may read it.
    """
    result: Dict[str, Any] = {
        "tmp_file": str(tmp_path),
        "tmp_size_mb": round(tmp_path.stat().st_size / (1024 * 1024), 3) if tmp_path.exists() else None,
        "tmp_mtime": datetime.fromtimestamp(tmp_path.stat().st_mtime).isoformat(timespec="seconds") if tmp_path.exists() else None,
        "tmp_rows": None,
        "processed_source_rows_est": None,
        "current_battery_id": None,
        "current_cycle": None,
        "current_phase": None,
        "tmp_read_error": None,
    }

    if not tmp_path.exists():
        return result

    wanted = [
        "feature_level",
        "battery_id",
        "cycle_index",
        "cycle_number",
        "phase",
        "phase_folder",
        "source_row_id",
        "dataset",
    ]

    cols_available = parquet_columns(tmp_path)
    if cols_available:
        cols = [c for c in wanted if c in cols_available]
    else:
        cols = wanted

    try:
        df = pd.read_parquet(tmp_path, engine="fastparquet", columns=cols)
    except Exception as exc:
        result["tmp_read_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["tmp_rows"] = int(len(df))

    if df.empty:
        return result

    if "feature_level" in df.columns:
        phase_rows = df[df["feature_level"].astype(str) == "phase"].copy()
    else:
        phase_rows = df.copy()

    if "source_row_id" in phase_rows.columns:
        src = pd.to_numeric(phase_rows["source_row_id"], errors="coerce")
        if src.notna().any():
            idx = src.idxmax()
            result["processed_source_rows_est"] = int(src.max()) + 1
            row = phase_rows.loc[idx]
        else:
            row = phase_rows.iloc[-1]
            result["processed_source_rows_est"] = int(len(phase_rows))
    else:
        row = phase_rows.iloc[-1]
        result["processed_source_rows_est"] = int(len(phase_rows))

    result["current_battery_id"] = row.get("battery_id", None)
    result["current_cycle"] = row.get("cycle_index", row.get("cycle_number", None))
    result["current_phase"] = row.get("phase", row.get("phase_folder", None))

    return result


def read_latest_progress_log(run_dir: Optional[Path], job_id: str) -> Dict[str, Any]:
    if run_dir is None:
        return {}

    progress_dir = run_dir / "progress_logs"
    if not progress_dir.exists():
        return {}

    matches = sorted(progress_dir.glob(f"*{job_id}*.progress.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return {}

    path = matches[0]
    last_line = None

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
    except Exception:
        return {"progress_log_path": str(path)}

    if not last_line:
        return {"progress_log_path": str(path)}

    try:
        data = json.loads(last_line)
    except Exception:
        data = {}

    data["progress_log_path"] = str(path)
    return data


def build_snapshot(jobs: pd.DataFrame, run_dir: Optional[Path]) -> pd.DataFrame:
    rows = []

    status_events = read_latest_status_events(run_dir)
    latest_status_by_job: Dict[str, Dict[str, Any]] = {}

    if not status_events.empty and "job_id" in status_events.columns:
        for _, event in status_events.iterrows():
            latest_status_by_job[str(event["job_id"])] = event.to_dict()

    for _, job in jobs.iterrows():
        output_path = Path(job["output_file"])
        done_path = Path(job["done_file"])
        error_path = Path(job["error_file"])

        done_info = read_json(done_path) if done_path.exists() else {}
        latest_event = latest_status_by_job.get(str(job["job_id"]), {})

        tmp_files = find_tmp_files(output_path)
        tmp_info = read_tmp_progress(tmp_files[0]) if tmp_files else {}

        progress_info = read_latest_progress_log(run_dir, str(job["job_id"]))

        if done_path.exists() and output_path.exists():
            status = "done"
        elif error_path.exists():
            status = "failed"
        elif tmp_files:
            status = "running_tmp"
        elif latest_event.get("status") == "skipped_done":
            status = "skipped_done"
        elif latest_event.get("status") == "failed":
            status = "failed"
        elif latest_event.get("status") == "done":
            status = "done_event_no_done"
        else:
            status = "pending_or_running"

        expected_rows = job.get("expected_input_rows")
        done_input_rows = done_info.get("input_rows", latest_event.get("input_rows"))
        done_output_rows = done_info.get("output_rows", latest_event.get("output_rows"))

        if status in {"done", "skipped_done", "done_event_no_done"}:
            processed_source_rows = done_input_rows if done_input_rows is not None else expected_rows
        elif progress_info.get("source_row_id") is not None:
            processed_source_rows = int(progress_info.get("source_row_id")) + 1
        elif tmp_info.get("processed_source_rows_est") is not None:
            processed_source_rows = tmp_info.get("processed_source_rows_est")
        else:
            processed_source_rows = 0

        if expected_rows and expected_rows > 0:
            file_progress_pct = 100.0 * float(processed_source_rows or 0) / float(expected_rows)
            file_progress_pct = max(0.0, min(100.0, file_progress_pct))
        else:
            file_progress_pct = None

        current_battery = (
            progress_info.get("battery_id")
            or tmp_info.get("current_battery_id")
        )
        current_cycle = (
            progress_info.get("cycle")
            or tmp_info.get("current_cycle")
        )
        current_phase = (
            progress_info.get("phase")
            or tmp_info.get("current_phase")
        )

        rows.append(
            {
                "job_id": job["job_id"],
                "dataset": job["dataset"],
                "phase_folder": job["phase_folder"],
                "input_name": job["input_name"],
                "status": status,
                "expected_input_rows": expected_rows,
                "processed_source_rows_est": processed_source_rows,
                "file_progress_pct": file_progress_pct,
                "done_input_rows": done_input_rows,
                "done_output_rows": done_output_rows,
                "current_battery_id": current_battery,
                "current_cycle": current_cycle,
                "current_phase": current_phase,
                "tmp_file": tmp_info.get("tmp_file"),
                "tmp_size_mb": tmp_info.get("tmp_size_mb"),
                "tmp_rows": tmp_info.get("tmp_rows"),
                "tmp_read_error": tmp_info.get("tmp_read_error"),
                "progress_log_path": progress_info.get("progress_log_path"),
                "finished_at": done_info.get("finished_at", latest_event.get("finished_at")),
                "input_file": job["input_file"],
                "output_file": job["output_file"],
                "done_file": job["done_file"],
                "error_file": job["error_file"],
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["phase_folder", "dataset", "input_name"]).reset_index(drop=True)
    return out


def summarize(snapshot: pd.DataFrame) -> Dict[str, Any]:
    total_files = len(snapshot)
    done_mask = snapshot["status"].isin(["done", "skipped_done", "done_event_no_done"])
    failed_mask = snapshot["status"].str.contains("failed", na=False)
    running_mask = snapshot["status"].isin(["running_tmp"])

    total_rows = pd.to_numeric(snapshot["expected_input_rows"], errors="coerce").fillna(0).sum()
    processed_rows = pd.to_numeric(snapshot["processed_source_rows_est"], errors="coerce").fillna(0).sum()

    return {
        "total_files": int(total_files),
        "done_files": int(done_mask.sum()),
        "running_files": int(running_mask.sum()),
        "failed_files": int(failed_mask.sum()),
        "pending_files": int(total_files - done_mask.sum() - running_mask.sum() - failed_mask.sum()),
        "file_progress_pct": round(100.0 * done_mask.sum() / max(total_files, 1), 2),
        "total_input_rows": int(total_rows),
        "processed_input_rows_est": int(processed_rows),
        "row_progress_pct": round(100.0 * processed_rows / max(total_rows, 1), 2),
    }


def clear_screen() -> None:
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        pass


def print_dashboard(snapshot: pd.DataFrame, summary: Dict[str, Any], run_dir: Optional[Path], monitor_dir: Path) -> None:
    print("Feature extraction v3 monitor")
    print("-" * 80)
    print(f"Time:                {now_text()}")
    print(f"Run dir:             {run_dir if run_dir else 'not found'}")
    print(f"Monitor snapshots:   {monitor_dir}")
    print("")
    print(f"Files done/running/pending/failed: "
          f"{summary['done_files']}/{summary['running_files']}/{summary['pending_files']}/{summary['failed_files']} "
          f"of {summary['total_files']}")
    print(f"File progress:       {summary['file_progress_pct']:.2f}%")
    print(f"Row progress est.:   {summary['row_progress_pct']:.2f}% "
          f"({summary['processed_input_rows_est']}/{summary['total_input_rows']} input rows)")
    print("")

    running = snapshot[snapshot["status"].isin(["running_tmp", "pending_or_running"])].copy()
    running = running.sort_values(["status", "phase_folder", "dataset"])

    if not running.empty:
        print("Running / not finished")
        print("-" * 80)
        cols = [
            "dataset", "phase_folder", "status", "file_progress_pct",
            "processed_source_rows_est", "expected_input_rows",
            "current_battery_id", "current_cycle", "input_name",
        ]
        display = running[cols].copy()
        display["file_progress_pct"] = display["file_progress_pct"].map(lambda x: "" if pd.isna(x) else f"{x:6.2f}%")
        display["current_battery_id"] = display["current_battery_id"].map(lambda x: "" if pd.isna(x) else compact(x, 32))
        display["input_name"] = display["input_name"].map(lambda x: compact(x, 40))
        print(display.to_string(index=False))
        print("")
    else:
        print("No running/pending files detected.")
        print("")

    done = snapshot[snapshot["status"].isin(["done", "skipped_done", "done_event_no_done"])].copy()
    if not done.empty:
        print("Completed files")
        print("-" * 80)
        cols = ["dataset", "phase_folder", "status", "done_input_rows", "done_output_rows", "finished_at", "input_name"]
        display = done[cols].copy()
        display["input_name"] = display["input_name"].map(lambda x: compact(x, 48))
        print(display.to_string(index=False))
        print("")

    failed = snapshot[snapshot["status"].str.contains("failed", na=False)].copy()
    if not failed.empty:
        print("Failed files")
        print("-" * 80)
        cols = ["dataset", "phase_folder", "status", "input_name", "error_file"]
        print(failed[cols].to_string(index=False))
        print("")


def write_snapshots(snapshot: pd.DataFrame, summary: Dict[str, Any], monitor_dir: Path) -> None:
    ensure_dir(monitor_dir)

    snapshot.to_csv(monitor_dir / "monitor_snapshot_latest.csv", index=False, encoding="utf-8-sig")

    done = snapshot[snapshot["status"].isin(["done", "skipped_done", "done_event_no_done"])]
    running = snapshot[snapshot["status"].isin(["running_tmp", "pending_or_running"])]
    failed = snapshot[snapshot["status"].str.contains("failed", na=False)]

    done.to_csv(monitor_dir / "monitor_completed_latest.csv", index=False, encoding="utf-8-sig")
    running.to_csv(monitor_dir / "monitor_running_or_pending_latest.csv", index=False, encoding="utf-8-sig")
    failed.to_csv(monitor_dir / "monitor_failed_latest.csv", index=False, encoding="utf-8-sig")

    summary_record = dict(summary)
    summary_record["timestamp"] = now_text()
    summary_path = monitor_dir / "monitor_summary_history.csv"

    pd.DataFrame([summary_record]).to_csv(
        summary_path,
        mode="a",
        header=not summary_path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor build_hierarchical_features_v3 execution progress.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--monitor-root", type=Path, default=DEFAULT_MONITOR_ROOT)
    parser.add_argument("--phase-dirs", nargs="*", default=list(PHASE_FOLDERS))
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--run-dir", type=Path, default=None, help="Specific extraction run directory. Default: latest run_* in reports-root.")
    parser.add_argument("--refresh-seconds", type=float, default=30.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    datasets = parse_dataset_args(args.dataset)
    monitor_dir = args.monitor_root / run_id()
    ensure_dir(monitor_dir)

    jobs = discover_jobs(
        input_root=args.input_root,
        output_root=args.output_root,
        phase_dirs=args.phase_dirs,
        datasets=datasets,
    )

    if jobs.empty:
        raise FileNotFoundError("No input Parquet jobs found in the selected phase folders.")

    while True:
        run_dir = args.run_dir if args.run_dir else latest_run_dir(args.reports_root)
        snapshot = build_snapshot(jobs, run_dir)
        summary = summarize(snapshot)

        write_snapshots(snapshot, summary, monitor_dir)

        if not args.no_clear:
            clear_screen()

        print_dashboard(snapshot, summary, run_dir, monitor_dir)

        if not args.watch:
            break

        if summary["done_files"] + summary["failed_files"] >= summary["total_files"]:
            print("All files are either completed or failed. Monitor finished.")
            break

        time.sleep(max(1.0, float(args.refresh_seconds)))


if __name__ == "__main__":
    main()
