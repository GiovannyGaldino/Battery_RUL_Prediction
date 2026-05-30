"""
check_feature_extraction_state_v3.py

Read-only state checker for hierarchical feature extraction v3.

Typical use from the repository root:

    python src/feature_extraction/check_feature_extraction_state_v3.py \
        --input-root data/interim \
        --output-root data/processed/features_hierarchical_v3 \
        --reports-root reports/feature_extraction_v3 \
        --state-root reports/feature_extraction_v3_state \
        --deep-check

For continuous monitoring, add:

    --watch

The script does not modify extraction outputs. It reads inputs, generated feature
files, .done markers, .error files, temporary files, status logs, and targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import fastparquet  # type: ignore
except Exception:  # pragma: no cover
    fastparquet = None

DEFAULT_INPUT_ROOT = Path("data/interim")
DEFAULT_OUTPUT_ROOT = Path("data/processed/features_hierarchical_v3")
DEFAULT_REPORTS_ROOT = Path("reports/feature_extraction_v3")
DEFAULT_STATE_ROOT = Path("reports/feature_extraction_v3_state")
PHASE_DIRS = ("charge", "discharge")

KEY_COLUMNS = [
    "dataset", "battery_id", "phase_folder", "cycle_index", "cycle_number",
    "phase", "feature_level", "stage_index", "stage_total", "source_row_id",
]

TARGET_COLUMNS = [
    "target_capacity_ah", "target_initial_capacity_ah", "target_capacity_ratio_to_initial",
    "target_soh_recomputed", "target_eol80_found", "target_rul80_steps_clipped",
    "target_eol70_found", "target_rul70_steps_clipped",
]

MIN_EXPECTED_FEATURE_COLUMNS = [
    "dataset", "battery_id", "phase_folder", "source_row_id", "feature_level",
    "target_capacity_ah", "target_soh_recomputed", "time_n_valid",
    "voltage_min", "voltage_max", "current_mean", "capacity_last",
]

STANDARD_INPUT_REQUIRED = [
    "dataset", "battery_id", "cycle_number", "cycle_index", "cycle_life",
    "remaining_cycles", "phase", "time_s", "voltage_v", "current_a",
    "temperature_c", "capacity_ah",
]


def now_id(prefix: str = "state") -> str:
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return str(value)


def short_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def infer_dataset_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"_(charge|discharge).*", "", stem)
    stem = stem.replace("nasa_pcoe", "pcoe").replace("nasa-random-walk", "nrw")
    return stem.upper() if stem else "UNKNOWN"


def parse_dataset_args(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(part.upper())
    return list(dict.fromkeys(out))


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_size_mb(path: Path) -> Optional[float]:
    try:
        return round(path.stat().st_size / (1024 * 1024), 3)
    except Exception:
        return None


def file_mtime(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return None


def parquet_columns(path: Path) -> List[str]:
    if not path.exists():
        return []
    if fastparquet is not None:
        try:
            return list(fastparquet.ParquetFile(str(path)).columns)
        except Exception:
            pass
    try:
        return list(pd.read_parquet(path, engine="fastparquet", columns=[]).columns)
    except Exception:
        return []


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


def read_parquet_selected_safe(path: Path, columns: Sequence[str]) -> Tuple[pd.DataFrame, str]:
    available = parquet_columns(path)
    selected = [c for c in columns if c in available]
    if not selected:
        return pd.DataFrame(), "no_selected_columns_available"
    try:
        return pd.read_parquet(path, engine="fastparquet", columns=selected), ""
    except Exception as exc:
        # Fallback: read column by column. This avoids failures with nullable integer columns.
        frames = []
        errors = []
        for col in selected:
            try:
                frames.append(pd.read_parquet(path, engine="fastparquet", columns=[col]))
            except Exception as col_exc:
                errors.append(f"{col}: {type(col_exc).__name__}: {col_exc}")
        if not frames:
            return pd.DataFrame(), f"{type(exc).__name__}: {exc}"
        return pd.concat(frames, axis=1).loc[:, lambda x: ~x.columns.duplicated()], "; ".join(errors)


def latest_run_dir(reports_root: Path) -> Optional[Path]:
    if not reports_root.exists():
        return None
    candidates = [p for p in reports_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_status_events(run_dir: Optional[Path]) -> pd.DataFrame:
    if run_dir is None:
        return pd.DataFrame()
    path = run_dir / "status.jsonl"
    if not path.exists():
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
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


def latest_event_by_job(run_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    events = read_status_events(run_dir)
    out: Dict[str, Dict[str, Any]] = {}
    if events.empty or "job_id" not in events.columns:
        return out
    for _, row in events.iterrows():
        out[str(row["job_id"])] = row.to_dict()
    return out


def discover_input_jobs(
    input_root: Path,
    output_root: Path,
    phase_dirs: Sequence[str],
    datasets: Sequence[str],
) -> pd.DataFrame:
    wanted = {d.upper() for d in datasets if d}
    records: List[Dict[str, Any]] = []

    for phase in phase_dirs:
        phase_dir = input_root / phase
        if not phase_dir.exists():
            records.append({
                "job_id": "",
                "dataset": "",
                "phase_folder": phase,
                "input_file": str(phase_dir),
                "input_name": phase_dir.name,
                "input_exists": False,
                "expected_input_rows": None,
                "input_missing_reason": "phase_folder_not_found",
            })
            continue

        for path in sorted(phase_dir.glob("*.parquet"), key=lambda p: str(p).lower()):
            dataset = infer_dataset_from_filename(path)
            if wanted and dataset.upper() not in wanted:
                continue
            job_id = short_hash(f"{phase}/{path.name}")
            out_dir = output_root / phase / dataset.lower()
            output_path = out_dir / f"{path.stem}__{job_id}.features.parquet"
            done_path = Path(str(output_path) + ".done")
            error_path = out_dir / f"{path.stem}__{job_id}.error.txt"

            records.append({
                "job_id": job_id,
                "dataset": dataset,
                "phase_folder": phase,
                "input_file": str(path),
                "input_name": path.name,
                "input_exists": True,
                "input_size_mb": file_size_mb(path),
                "input_modified_at": file_mtime(path),
                "expected_input_rows": parquet_row_count(path),
                "input_columns_count": len(parquet_columns(path)),
                "output_file": str(output_path),
                "done_file": str(done_path),
                "error_file": str(error_path),
            })

    return pd.DataFrame(records)


def discover_adapter_clean_files(input_root: Path, datasets: Sequence[str]) -> pd.DataFrame:
    """
    Checks the upstream adapter outputs in data/interim/<dataset>/clean/*.parquet.
    This is useful to identify whether missing feature jobs are caused by missing merge files.
    """
    wanted = {d.upper() for d in datasets if d}
    records: List[Dict[str, Any]] = []
    if not input_root.exists():
        return pd.DataFrame()

    for dataset_dir in sorted([p for p in input_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        if dataset_dir.name.lower() in {"charge", "discharge"}:
            continue
        dataset = dataset_dir.name.upper()
        if wanted and dataset not in wanted:
            continue
        clean_dir = dataset_dir / "clean"
        for phase in PHASE_DIRS:
            matches = sorted(clean_dir.glob(f"*_{phase}_clean_cycles.parquet")) if clean_dir.exists() else []
            if not matches:
                records.append({
                    "dataset_folder": dataset_dir.name,
                    "phase": phase,
                    "clean_file_found": False,
                    "clean_file": "",
                    "rows": None,
                    "size_mb": None,
                    "missing_reason": "not_found",
                })
            else:
                for path in matches:
                    records.append({
                        "dataset_folder": dataset_dir.name,
                        "phase": phase,
                        "clean_file_found": True,
                        "clean_file": str(path),
                        "rows": parquet_row_count(path),
                        "size_mb": file_size_mb(path),
                        "modified_at": file_mtime(path),
                        "columns_count": len(parquet_columns(path)),
                        "missing_required_columns": ", ".join([c for c in STANDARD_INPUT_REQUIRED if c not in parquet_columns(path)]),
                    })
    return pd.DataFrame(records)


def find_tmp_files(output_path: Path) -> List[Path]:
    if not output_path.parent.exists():
        return []
    return sorted(output_path.parent.glob(f".{output_path.name}.tmp.*"), key=lambda p: p.stat().st_mtime, reverse=True)


def tmp_is_stale(path: Path, stale_minutes: float) -> bool:
    try:
        age_s = time.time() - path.stat().st_mtime
        return age_s > stale_minutes * 60.0
    except Exception:
        return False


def inspect_feature_file(path: Path, expected_input_rows: Optional[int], deep_check: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "output_exists": path.exists(),
        "output_size_mb": file_size_mb(path) if path.exists() else None,
        "output_modified_at": file_mtime(path) if path.exists() else None,
        "output_rows_meta": parquet_row_count(path) if path.exists() else None,
        "feature_columns_count": len(parquet_columns(path)) if path.exists() else None,
        "missing_min_feature_columns": "",
        "feature_read_error": "",
        "phase_rows": None,
        "stage_rows": None,
        "feature_levels": "",
        "phase_rows_match_input": None,
        "source_row_id_max": None,
        "battery_count_output": None,
    }
    if not path.exists():
        return out

    cols = parquet_columns(path)
    out["missing_min_feature_columns"] = ", ".join([c for c in MIN_EXPECTED_FEATURE_COLUMNS if c not in cols])

    if not deep_check:
        return out

    df, err = read_parquet_selected_safe(path, KEY_COLUMNS + TARGET_COLUMNS)
    out["feature_read_error"] = err
    if df.empty:
        return out

    if "feature_level" in df.columns:
        vc = df["feature_level"].astype(str).value_counts(dropna=False)
        out["feature_levels"] = "; ".join([f"{k}={int(v)}" for k, v in vc.items()])
        out["phase_rows"] = int(vc.get("phase", 0))
        out["stage_rows"] = int(vc.get("stage", 0))
    else:
        out["phase_rows"] = None
        out["stage_rows"] = None

    if "source_row_id" in df.columns:
        source_ids = pd.to_numeric(df["source_row_id"], errors="coerce")
        if source_ids.notna().any():
            out["source_row_id_max"] = int(source_ids.max())

    if "battery_id" in df.columns:
        out["battery_count_output"] = int(df["battery_id"].nunique(dropna=True))

    if expected_input_rows is not None and out.get("phase_rows") is not None:
        try:
            out["phase_rows_match_input"] = bool(int(out["phase_rows"]) == int(expected_input_rows))
        except Exception:
            out["phase_rows_match_input"] = None

    return out


def estimate_tmp_progress(tmp_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "tmp_file": str(tmp_path),
        "tmp_size_mb": file_size_mb(tmp_path),
        "tmp_modified_at": file_mtime(tmp_path),
        "tmp_rows_meta": parquet_row_count(tmp_path),
        "tmp_progress_rows_est": None,
        "tmp_current_battery_id": None,
        "tmp_current_cycle": None,
        "tmp_read_error": "",
    }
    if not tmp_path.exists():
        return out
    df, err = read_parquet_selected_safe(tmp_path, ["feature_level", "source_row_id", "battery_id", "cycle_index", "cycle_number", "phase"])
    out["tmp_read_error"] = err
    if df.empty:
        return out
    if "feature_level" in df.columns:
        df2 = df[df["feature_level"].astype(str) == "phase"].copy()
        if df2.empty:
            df2 = df.copy()
    else:
        df2 = df.copy()
    if "source_row_id" in df2.columns:
        src = pd.to_numeric(df2["source_row_id"], errors="coerce")
        if src.notna().any():
            idx = src.idxmax()
            out["tmp_progress_rows_est"] = int(src.max()) + 1
            row = df2.loc[idx]
        else:
            row = df2.iloc[-1]
            out["tmp_progress_rows_est"] = int(len(df2))
    else:
        row = df2.iloc[-1]
        out["tmp_progress_rows_est"] = int(len(df2))
    out["tmp_current_battery_id"] = safe_str(row.get("battery_id", ""))
    out["tmp_current_cycle"] = safe_str(row.get("cycle_index", row.get("cycle_number", "")))
    return out


def build_state_table(
    jobs: pd.DataFrame,
    run_dir: Optional[Path],
    deep_check: bool,
    stale_minutes: float,
) -> pd.DataFrame:
    latest_events = latest_event_by_job(run_dir)
    rows: List[Dict[str, Any]] = []

    for _, job in jobs.iterrows():
        if not bool(job.get("input_exists", False)):
            rows.append(job.to_dict())
            continue

        output_path = Path(str(job["output_file"]))
        done_path = Path(str(job["done_file"]))
        error_path = Path(str(job["error_file"]))
        tmp_files = find_tmp_files(output_path)
        latest_tmp = tmp_files[0] if tmp_files else None
        event = latest_events.get(str(job["job_id"]), {})
        done_payload = read_json(done_path) if done_path.exists() else {}

        feature_info = inspect_feature_file(output_path, job.get("expected_input_rows"), deep_check=deep_check)
        tmp_info = estimate_tmp_progress(latest_tmp) if latest_tmp else {}

        output_exists = bool(feature_info.get("output_exists"))
        done_exists = done_path.exists()
        error_exists = error_path.exists()
        tmp_exists = bool(tmp_files)
        output_rows_meta = feature_info.get("output_rows_meta")
        output_readable = output_exists and output_rows_meta is not None

        if done_exists and output_readable:
            status = "done_valid"
        elif done_exists and output_exists and not output_readable:
            status = "done_output_unreadable"
        elif error_exists:
            status = "failed"
        elif tmp_exists and latest_tmp is not None and tmp_is_stale(latest_tmp, stale_minutes):
            status = "stale_tmp"
        elif tmp_exists:
            status = "running_tmp"
        elif output_exists and not done_exists:
            status = "output_without_done"
        elif event.get("status") == "done":
            status = "done_event_no_marker"
        elif event.get("status") == "skipped_done":
            status = "skipped_done_event_no_marker"
        elif event.get("status") == "failed":
            status = "failed_event_no_error_file"
        else:
            status = "pending"

        expected = job.get("expected_input_rows")
        processed = 0
        if status in {"done_valid", "done_output_unreadable", "done_event_no_marker", "skipped_done_event_no_marker"}:
            processed = done_payload.get("input_rows") or event.get("input_rows") or expected or 0
        elif status in {"running_tmp", "stale_tmp"}:
            processed = tmp_info.get("tmp_progress_rows_est") or 0
        elif status == "output_without_done" and feature_info.get("phase_rows_match_input") is True:
            processed = expected or 0
        elif status == "output_without_done" and feature_info.get("source_row_id_max") is not None:
            processed = int(feature_info.get("source_row_id_max")) + 1

        file_progress_pct = None
        try:
            if expected and float(expected) > 0:
                file_progress_pct = max(0.0, min(100.0, 100.0 * float(processed) / float(expected)))
        except Exception:
            pass

        issues: List[str] = []
        if error_exists or "failed" in status:
            issues.append("failed_job")
        if status == "stale_tmp":
            issues.append("stale_tmp_check_if_process_stopped")
        if status == "output_without_done":
            issues.append("output_exists_but_done_marker_missing")
        if status == "done_output_unreadable":
            issues.append("done_marker_but_output_unreadable")
        if feature_info.get("missing_min_feature_columns"):
            issues.append("missing_minimum_feature_columns")
        if deep_check and feature_info.get("phase_rows_match_input") is False:
            issues.append("phase_rows_do_not_match_input_rows")
        if feature_info.get("feature_read_error"):
            issues.append("feature_read_warning")

        row = job.to_dict()
        row.update({
            "status": status,
            "processed_input_rows_est": int(processed or 0),
            "file_progress_pct": file_progress_pct,
            "done_exists": done_exists,
            "error_exists": error_exists,
            "tmp_exists": tmp_exists,
            "tmp_count": len(tmp_files),
            "latest_event_status": event.get("status", ""),
            "latest_event_finished_at": event.get("finished_at", ""),
            "done_finished_at": done_payload.get("finished_at", ""),
            "done_input_rows": done_payload.get("input_rows", event.get("input_rows")),
            "done_output_rows": done_payload.get("output_rows", event.get("output_rows")),
            "error_preview": "",
            "issues": "; ".join(issues),
        })
        if error_exists:
            try:
                row["error_preview"] = error_path.read_text(encoding="utf-8", errors="ignore")[:1000]
            except Exception:
                row["error_preview"] = "could_not_read_error_file"
        row.update(feature_info)
        row.update(tmp_info)
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty and "dataset" in out.columns:
        out = out.sort_values(["phase_folder", "dataset", "input_name"], na_position="last").reset_index(drop=True)
    return out


def summarize_state(state: pd.DataFrame) -> Dict[str, Any]:
    if state.empty:
        return {
            "generated_at": now_iso(),
            "total_jobs": 0,
            "done_jobs": 0,
            "failed_jobs": 0,
            "running_jobs": 0,
            "pending_jobs": 0,
            "stale_tmp_jobs": 0,
            "file_progress_pct": 0.0,
            "row_progress_pct": 0.0,
        }

    status = state.get("status", pd.Series(dtype=str)).astype(str)
    done_mask = status.isin(["done_valid", "done_event_no_marker", "skipped_done_event_no_marker"])
    failed_mask = status.str.contains("failed", na=False)
    running_mask = status.isin(["running_tmp"])
    stale_mask = status.isin(["stale_tmp"])
    pending_mask = status.isin(["pending", "output_without_done", "done_output_unreadable"])

    expected = pd.to_numeric(state.get("expected_input_rows", pd.Series(dtype=float)), errors="coerce").fillna(0)
    processed = pd.to_numeric(state.get("processed_input_rows_est", pd.Series(dtype=float)), errors="coerce").fillna(0)

    return {
        "generated_at": now_iso(),
        "total_jobs": int(len(state)),
        "done_jobs": int(done_mask.sum()),
        "failed_jobs": int(failed_mask.sum()),
        "running_jobs": int(running_mask.sum()),
        "pending_jobs": int(pending_mask.sum()),
        "stale_tmp_jobs": int(stale_mask.sum()),
        "jobs_with_issues": int(state.get("issues", pd.Series(dtype=str)).astype(str).ne("").sum()),
        "file_progress_pct": round(100.0 * float(done_mask.sum()) / max(len(state), 1), 2),
        "total_input_rows": int(expected.sum()),
        "processed_input_rows_est": int(processed.sum()),
        "row_progress_pct": round(100.0 * float(processed.sum()) / max(float(expected.sum()), 1.0), 2),
    }


def summarize_by_dataset_phase(state: pd.DataFrame) -> pd.DataFrame:
    if state.empty or "dataset" not in state.columns:
        return pd.DataFrame()
    df = state.copy()
    if "status" not in df.columns:
        df["status"] = "unknown"
    if "issues" not in df.columns:
        df["issues"] = ""
    for col in ["expected_input_rows", "processed_input_rows_est", "output_rows_meta", "phase_rows", "stage_rows"]:
        if col not in df.columns:
            df[col] = 0
    df["is_done"] = df["status"].astype(str).isin(["done_valid", "done_event_no_marker", "skipped_done_event_no_marker"])
    df["is_failed"] = df["status"].astype(str).str.contains("failed", na=False)
    df["is_running"] = df["status"].astype(str).isin(["running_tmp", "stale_tmp"])
    df["has_issues"] = df["issues"].astype(str).ne("")
    df["expected_input_rows"] = pd.to_numeric(df["expected_input_rows"], errors="coerce").fillna(0)
    df["processed_input_rows_est"] = pd.to_numeric(df["processed_input_rows_est"], errors="coerce").fillna(0)
    for col in ["output_rows_meta", "phase_rows", "stage_rows"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    grouped = df.groupby(["dataset", "phase_folder"], dropna=False).agg(
        files=("job_id", "count"),
        done_files=("is_done", "sum"),
        failed_files=("is_failed", "sum"),
        running_or_stale_files=("is_running", "sum"),
        files_with_issues=("has_issues", "sum"),
        input_rows=("expected_input_rows", "sum"),
        processed_rows_est=("processed_input_rows_est", "sum"),
        output_rows_meta=("output_rows_meta", "sum"),
        phase_rows=("phase_rows", "sum"),
        stage_rows=("stage_rows", "sum"),
    ).reset_index()
    grouped["file_progress_pct"] = (100 * grouped["done_files"] / grouped["files"].clip(lower=1)).round(2)
    grouped["row_progress_pct"] = (100 * grouped["processed_rows_est"] / grouped["input_rows"].clip(lower=1)).round(2)
    return grouped.sort_values(["phase_folder", "dataset"]).reset_index(drop=True)


def inspect_targets(output_root: Path, reports_root: Path, run_dir: Optional[Path]) -> Dict[str, Any]:
    target_path = output_root / "_targets" / "targets_v3.parquet"
    cap_summary_out = output_root / "_targets" / "capacity_life_summary_v3.csv"
    out: Dict[str, Any] = {
        "targets_path": str(target_path),
        "targets_exists": target_path.exists(),
        "targets_rows": parquet_row_count(target_path) if target_path.exists() else None,
        "targets_size_mb": file_size_mb(target_path) if target_path.exists() else None,
        "targets_modified_at": file_mtime(target_path) if target_path.exists() else None,
        "capacity_summary_output_path": str(cap_summary_out),
        "capacity_summary_output_exists": cap_summary_out.exists(),
        "capacity_summary_output_rows": None,
        "latest_run_dir": str(run_dir) if run_dir else "",
    }
    if cap_summary_out.exists():
        try:
            out["capacity_summary_output_rows"] = int(len(pd.read_csv(cap_summary_out)))
        except Exception:
            out["capacity_summary_output_rows"] = None
    if run_dir:
        for name in ["metadata.json", "dataset_phase_target_summary.csv", "capacity_life_summary_v3.csv", "summary.csv", "failures.csv"]:
            p = run_dir / name
            out[f"run_{name}_exists"] = p.exists()
            out[f"run_{name}_path"] = str(p)
    return out


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No records._"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def build_markdown_report(
    summary: Dict[str, Any],
    state: pd.DataFrame,
    by_dataset_phase: pd.DataFrame,
    targets_state: Dict[str, Any],
    adapter_state: pd.DataFrame,
    run_dir: Optional[Path],
) -> str:
    lines: List[str] = []
    lines.append("# Feature Extraction v3 — State Report")
    lines.append("")
    lines.append(f"Generated at: `{summary.get('generated_at', now_iso())}`")
    lines.append(f"Latest extraction run: `{run_dir if run_dir else 'not found'}`")
    lines.append("")
    lines.append("## Overall progress")
    lines.append("")
    lines.append(f"- Jobs: **{summary['done_jobs']} done**, **{summary['running_jobs']} running**, **{summary['stale_tmp_jobs']} stale tmp**, **{summary['pending_jobs']} pending**, **{summary['failed_jobs']} failed**, total **{summary['total_jobs']}**.")
    lines.append(f"- File progress: **{summary['file_progress_pct']}%**.")
    lines.append(f"- Row progress estimate: **{summary['row_progress_pct']}%** ({summary.get('processed_input_rows_est', 0)}/{summary.get('total_input_rows', 0)} input rows).")
    lines.append(f"- Jobs with warnings/issues: **{summary.get('jobs_with_issues', 0)}**.")
    lines.append("")
    lines.append("## Targets and capacity-life files")
    lines.append("")
    for key, value in targets_state.items():
        if key.endswith("_path") or key == "latest_run_dir":
            continue
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Summary by dataset and phase")
    lines.append("")
    keep = [c for c in ["dataset", "phase_folder", "files", "done_files", "failed_files", "running_or_stale_files", "files_with_issues", "input_rows", "processed_rows_est", "row_progress_pct", "phase_rows", "stage_rows"] if c in by_dataset_phase.columns]
    lines.append(markdown_table(by_dataset_phase[keep] if keep else by_dataset_phase, max_rows=60))
    lines.append("")

    issue_df = state[state.get("issues", pd.Series(dtype=str)).astype(str).ne("")].copy() if not state.empty else pd.DataFrame()
    lines.append("## Jobs with issues")
    lines.append("")
    if issue_df.empty:
        lines.append("_No issues detected._")
    else:
        cols = [c for c in ["dataset", "phase_folder", "status", "input_name", "file_progress_pct", "issues", "error_preview"] if c in issue_df.columns]
        lines.append(markdown_table(issue_df[cols], max_rows=80))
    lines.append("")

    unfinished = state[~state.get("status", pd.Series(dtype=str)).astype(str).isin(["done_valid", "done_event_no_marker", "skipped_done_event_no_marker"])] if not state.empty else pd.DataFrame()
    lines.append("## Unfinished jobs")
    lines.append("")
    if unfinished.empty:
        lines.append("_All jobs appear completed._")
    else:
        cols = [c for c in ["dataset", "phase_folder", "status", "input_name", "processed_input_rows_est", "expected_input_rows", "file_progress_pct", "tmp_current_battery_id", "tmp_current_cycle"] if c in unfinished.columns]
        lines.append(markdown_table(unfinished[cols], max_rows=80))
    lines.append("")

    if not adapter_state.empty:
        lines.append("## Upstream adapter clean files")
        lines.append("")
        missing = adapter_state[adapter_state.get("clean_file_found", pd.Series(dtype=bool)) == False].copy()  # noqa: E712
        bad_columns = adapter_state[adapter_state.get("missing_required_columns", pd.Series(dtype=str)).fillna("").astype(str).ne("")].copy()
        if missing.empty and bad_columns.empty:
            lines.append("_No missing clean adapter files or required-column gaps detected in per-dataset clean folders._")
        else:
            cols = [c for c in ["dataset_folder", "phase", "clean_file_found", "rows", "missing_required_columns", "missing_reason"] if c in adapter_state.columns]
            lines.append(markdown_table(pd.concat([missing, bad_columns], ignore_index=True)[cols].drop_duplicates(), max_rows=80))
        lines.append("")

    lines.append("## Generated files")
    lines.append("")
    lines.append("- `job_state.csv`: detailed state of every expected extraction job.")
    lines.append("- `summary_by_dataset_phase.csv`: progress grouped by dataset and phase.")
    lines.append("- `summary.json`: overall counters.")
    lines.append("- `targets_state.json`: target table and capacity-life summary status.")
    lines.append("- `adapter_clean_state.csv`: upstream adapter clean files found under `data/interim/<dataset>/clean`.")
    lines.append("")
    return "\n".join(lines)


def print_dashboard(summary: Dict[str, Any], by_dataset_phase: pd.DataFrame, state: pd.DataFrame, out_dir: Path) -> None:
    print("Feature extraction v3 state pipeline")
    print("-" * 90)
    print(f"Time:                  {now_text()}")
    print(f"Report dir:            {out_dir}")
    print(f"Jobs done/running/stale/pending/failed: "
          f"{summary['done_jobs']}/{summary['running_jobs']}/{summary['stale_tmp_jobs']}/{summary['pending_jobs']}/{summary['failed_jobs']} "
          f"of {summary['total_jobs']}")
    print(f"File progress:         {summary['file_progress_pct']:.2f}%")
    print(f"Row progress estimate: {summary['row_progress_pct']:.2f}% "
          f"({summary.get('processed_input_rows_est', 0)}/{summary.get('total_input_rows', 0)})")
    print(f"Jobs with issues:      {summary.get('jobs_with_issues', 0)}")
    print("")

    if not by_dataset_phase.empty:
        cols = [c for c in ["dataset", "phase_folder", "files", "done_files", "failed_files", "running_or_stale_files", "input_rows", "processed_rows_est", "row_progress_pct"] if c in by_dataset_phase.columns]
        print("By dataset/phase")
        print("-" * 90)
        print(by_dataset_phase[cols].to_string(index=False))
        print("")

    issues = state[state.get("issues", pd.Series(dtype=str)).astype(str).ne("")].copy() if not state.empty else pd.DataFrame()
    if not issues.empty:
        print("Issues")
        print("-" * 90)
        cols = [c for c in ["dataset", "phase_folder", "status", "input_name", "issues"] if c in issues.columns]
        print(issues[cols].to_string(index=False))
        print("")


def run_once(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    datasets = parse_dataset_args(args.dataset)
    run_dir = args.run_dir if args.run_dir else latest_run_dir(args.reports_root)

    jobs = discover_input_jobs(
        input_root=args.input_root,
        output_root=args.output_root,
        phase_dirs=args.phase_dirs,
        datasets=datasets,
    )

    state = build_state_table(
        jobs=jobs,
        run_dir=run_dir,
        deep_check=bool(args.deep_check),
        stale_minutes=float(args.stale_minutes),
    )

    by_dataset_phase = summarize_by_dataset_phase(state)
    summary = summarize_state(state)
    targets_state = inspect_targets(args.output_root, args.reports_root, run_dir)
    adapter_state = discover_adapter_clean_files(args.input_root, datasets)

    ensure_dir(out_dir)
    write_csv(jobs, out_dir / "expected_jobs.csv")
    write_csv(state, out_dir / "job_state.csv")
    write_csv(by_dataset_phase, out_dir / "summary_by_dataset_phase.csv")
    write_json(summary, out_dir / "summary.json")
    write_json(targets_state, out_dir / "targets_state.json")
    write_csv(adapter_state, out_dir / "adapter_clean_state.csv")

    report_md = build_markdown_report(
        summary=summary,
        state=state,
        by_dataset_phase=by_dataset_phase,
        targets_state=targets_state,
        adapter_state=adapter_state,
        run_dir=run_dir,
    )
    (out_dir / "state_report.md").write_text(report_md, encoding="utf-8")

    print_dashboard(summary, by_dataset_phase, state, out_dir)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verifica o estado atual da extração de características v3.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--phase-dirs", nargs="*", default=list(PHASE_DIRS))
    parser.add_argument("--dataset", action="append", default=None, help="Filtrar uma ou mais bases. Ex.: --dataset CALCE --dataset HUST")
    parser.add_argument("--run-dir", type=Path, default=None, help="Diretório run_* específico. Se omitido, usa o run_* mais recente.")
    parser.add_argument("--deep-check", action="store_true", help="Lê colunas-chave dos arquivos de features para verificar phase/stage rows e integridade lógica.")
    parser.add_argument("--stale-minutes", type=float, default=30.0, help="Tempo para classificar temporário sem atualização como stale_tmp.")
    parser.add_argument("--watch", action="store_true", help="Atualiza continuamente até todos os jobs terminarem ou falharem.")
    parser.add_argument("--refresh-seconds", type=float, default=30.0)
    parser.add_argument("--no-clear", action="store_true")
    return parser


def clear_screen() -> None:
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        pass


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    out_dir = args.state_root / now_id()
    ensure_dir(out_dir)

    while True:
        if not args.no_clear:
            clear_screen()
        summary = run_once(args, out_dir)

        if not args.watch:
            break

        finished = summary["done_jobs"] + summary["failed_jobs"]
        total = summary["total_jobs"]
        if total > 0 and finished >= total:
            print("All jobs are done or failed. State watch finished.")
            break

        time.sleep(max(1.0, float(args.refresh_seconds)))


if __name__ == "__main__":
    main()
