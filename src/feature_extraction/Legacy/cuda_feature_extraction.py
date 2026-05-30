import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import warnings
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT_DIR = r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries"

INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORT_DIR = os.path.join(ROOT_DIR, "data", "reports", "feature_extraction")

PHASES = ["charge", "discharge"]
P_POINTS = 15
PARQUET_ENGINE = "fastparquet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return np.nan


def to_tensor(values):
    try:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return None
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE)
    except Exception:
        return None


def finite_clean(t, v, i, temp, q):
    mask = torch.isfinite(t) & torch.isfinite(v) & torch.isfinite(q)

    if i is not None:
        mask = mask & torch.isfinite(i)

    if temp is not None:
        mask = mask & torch.isfinite(temp)

    t = t[mask]
    v = v[mask]
    q = q[mask]

    if i is not None:
        i = i[mask]
    else:
        i = torch.full_like(t, float("nan"))

    if temp is not None:
        temp = temp[mask]
    else:
        temp = torch.full_like(t, float("nan"))

    return t, v, i, temp, q


def tensor_stats(x, prefix):
    out = {}
    finite = x[torch.isfinite(x)]

    names = [
        "mean", "min", "max", "std", "var", "p05", "p25", "p50",
        "p75", "p95", "range", "skew", "kurtosis"
    ]

    if finite.numel() < 2:
        for name in names:
            out[f"{prefix}_{name}"] = np.nan
        return out

    mean = finite.mean()
    std = finite.std(unbiased=False)
    centered = finite - mean

    out[f"{prefix}_mean"] = safe_float(mean)
    out[f"{prefix}_min"] = safe_float(finite.min())
    out[f"{prefix}_max"] = safe_float(finite.max())
    out[f"{prefix}_std"] = safe_float(std)
    out[f"{prefix}_var"] = safe_float(finite.var(unbiased=False))
    out[f"{prefix}_p05"] = safe_float(torch.quantile(finite, 0.05))
    out[f"{prefix}_p25"] = safe_float(torch.quantile(finite, 0.25))
    out[f"{prefix}_p50"] = safe_float(torch.quantile(finite, 0.50))
    out[f"{prefix}_p75"] = safe_float(torch.quantile(finite, 0.75))
    out[f"{prefix}_p95"] = safe_float(torch.quantile(finite, 0.95))
    out[f"{prefix}_range"] = safe_float(finite.max() - finite.min())

    if safe_float(std) < 1e-10:
        out[f"{prefix}_skew"] = 0.0
        out[f"{prefix}_kurtosis"] = 0.0
    else:
        z = centered / (std + 1e-12)
        out[f"{prefix}_skew"] = safe_float(torch.mean(z ** 3))
        out[f"{prefix}_kurtosis"] = safe_float(torch.mean(z ** 4) - 3.0)

    return out


def gpu_interp_1d(x, y, x_new):
    order = torch.argsort(x)
    x = x[order]
    y = y[order]

    keep = torch.ones_like(x, dtype=torch.bool)
    keep[1:] = torch.diff(x) > 1e-8

    x = x[keep]
    y = y[keep]

    if x.numel() < 2:
        return torch.full_like(x_new, float("nan"))

    x_new = torch.clamp(x_new, x.min(), x.max())

    idx = torch.searchsorted(x, x_new, right=False)
    idx = torch.clamp(idx, 1, x.numel() - 1)

    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]

    weight = (x_new - x0) / (x1 - x0 + 1e-12)
    return y0 + weight * (y1 - y0)


def curve_features(v, q, phase, p=P_POINTS):
    out = {}
    tag = "CH" if phase == "charge" else "DI"

    try:
        v_min = v.min()
        v_max = v.max()

        if safe_float(v_max - v_min) <= 1e-8:
            raise ValueError("Voltage range too small.")

        v_grid = torch.linspace(v_min, v_max, p, device=DEVICE)
        q_grid = gpu_interp_1d(v, q, v_grid)

        for k in range(p):
            out[f"{tag}_{k + 1}"] = safe_float(q_grid[k])

        dq = torch.diff(q_grid)
        dv = torch.diff(v_grid)
        dqdv = dq / (dv + 1e-12)

        out[f"{tag}_area"] = safe_float(torch.trapz(q_grid, v_grid))
        out[f"{tag}_slope_mean"] = safe_float(dqdv.mean())
        out[f"{tag}_slope_std"] = safe_float(dqdv.std(unbiased=False))
        out[f"{tag}_slope_min"] = safe_float(dqdv.min())
        out[f"{tag}_slope_max"] = safe_float(dqdv.max())
        out[f"{tag}_range"] = safe_float(q_grid.max() - q_grid.min())
        out[f"{tag}_distance_to_initial"] = safe_float(torch.linalg.norm(q_grid - q_grid[0]))

        abs_dqdv = torch.abs(dqdv)
        peak_idx = torch.argmax(abs_dqdv)

        out[f"{tag}_dQdV_abs_peak"] = safe_float(abs_dqdv[peak_idx])
        out[f"{tag}_dQdV_peak_voltage"] = safe_float(v_grid[:-1][peak_idx])
        out[f"{tag}_dQdV_mean_abs"] = safe_float(abs_dqdv.mean())

    except Exception:
        for k in range(p):
            out[f"{tag}_{k + 1}"] = np.nan

        for name in [
            "area", "slope_mean", "slope_std", "slope_min", "slope_max",
            "range", "distance_to_initial", "dQdV_abs_peak",
            "dQdV_peak_voltage", "dQdV_mean_abs"
        ]:
            out[f"{tag}_{name}"] = np.nan

    return out


def time_gradient_features(t, x, prefix):
    out = {}

    try:
        if t.numel() < 3:
            raise ValueError("Too few points.")

        dt = torch.diff(t)
        dx = torch.diff(x)

        valid = torch.isfinite(dt) & torch.isfinite(dx) & (torch.abs(dt) > 1e-12)
        grad = dx[valid] / dt[valid]

        if grad.numel() < 2:
            raise ValueError("Invalid gradient.")

        out[f"{prefix}_dt_mean"] = safe_float(grad.mean())
        out[f"{prefix}_dt_std"] = safe_float(grad.std(unbiased=False))
        out[f"{prefix}_dt_min"] = safe_float(grad.min())
        out[f"{prefix}_dt_max"] = safe_float(grad.max())
        out[f"{prefix}_dt_abs_mean"] = safe_float(torch.abs(grad).mean())

    except Exception:
        out[f"{prefix}_dt_mean"] = np.nan
        out[f"{prefix}_dt_std"] = np.nan
        out[f"{prefix}_dt_min"] = np.nan
        out[f"{prefix}_dt_max"] = np.nan
        out[f"{prefix}_dt_abs_mean"] = np.nan

    return out


def segment_cc_cv_features(t, v, i, phase):
    out = {}

    try:
        i_abs = torch.abs(i)
        finite_i = i_abs[torch.isfinite(i_abs)]
        finite_v = v[torch.isfinite(v)]

        if finite_i.numel() < 10 or finite_v.numel() < 10:
            raise ValueError("Invalid current or voltage.")

        i_ref = torch.quantile(finite_i, 0.90)
        v_ref = torch.quantile(finite_v, 0.95)

        cc_mask = i_abs >= 0.85 * i_ref
        cv_mask = v >= 0.98 * v_ref

        total_duration = t[-1] - t[0]

        cc_duration = torch.sum(torch.diff(t) * cc_mask[:-1].float())
        cv_duration = torch.sum(torch.diff(t) * cv_mask[:-1].float())

        out[f"{phase}_cc_duration_s"] = safe_float(cc_duration)
        out[f"{phase}_cv_duration_s"] = safe_float(cv_duration)
        out[f"{phase}_cc_ratio"] = safe_float(cc_duration / (total_duration + 1e-12))
        out[f"{phase}_cv_ratio"] = safe_float(cv_duration / (total_duration + 1e-12))

        if cc_mask.sum() >= 5:
            t_cc = t[cc_mask]
            v_cc = v[cc_mask]
            A = torch.stack([t_cc, torch.ones_like(t_cc)], dim=1)
            sol = torch.linalg.lstsq(A, v_cc.unsqueeze(1)).solution.squeeze()
            out[f"{phase}_cc_voltage_slope"] = safe_float(sol[0])
        else:
            out[f"{phase}_cc_voltage_slope"] = np.nan

        if cv_mask.sum() >= 5:
            t_cv = t[cv_mask]
            i_cv = i[cv_mask]
            A = torch.stack([t_cv, torch.ones_like(t_cv)], dim=1)
            sol = torch.linalg.lstsq(A, i_cv.unsqueeze(1)).solution.squeeze()
            out[f"{phase}_cv_current_slope"] = safe_float(sol[0])
        else:
            out[f"{phase}_cv_current_slope"] = np.nan

    except Exception:
        out[f"{phase}_cc_duration_s"] = np.nan
        out[f"{phase}_cv_duration_s"] = np.nan
        out[f"{phase}_cc_ratio"] = np.nan
        out[f"{phase}_cv_ratio"] = np.nan
        out[f"{phase}_cc_voltage_slope"] = np.nan
        out[f"{phase}_cv_current_slope"] = np.nan

    return out


def extract_one_cycle(row, phase, prev_capacity=None):
    cycle = row.to_dict()
    out = {}

    t = to_tensor(cycle.get("time_s"))
    v = to_tensor(cycle.get("voltage_v"))
    i = to_tensor(cycle.get("current_a"))
    temp = to_tensor(cycle.get("temperature_c"))
    q = to_tensor(cycle.get("capacity_ah"))

    if t is None or v is None or q is None:
        raise ValueError("Missing mandatory vector: time_s, voltage_v or capacity_ah.")

    t, v, i, temp, q = finite_clean(t, v, i, temp, q)

    if t.numel() < 20:
        raise ValueError("Too few valid points after cleaning.")

    t = t - t[0]

    out.update(curve_features(v, q, phase, P_POINTS))

    out.update(tensor_stats(v, f"{phase}_voltage"))
    out.update(tensor_stats(i, f"{phase}_current"))
    out.update(tensor_stats(temp, f"{phase}_temperature"))
    out.update(tensor_stats(q, f"{phase}_capacity_curve"))

    out.update(time_gradient_features(t, v, f"{phase}_voltage"))
    out.update(time_gradient_features(t, i, f"{phase}_current"))
    out.update(time_gradient_features(t, temp, f"{phase}_temperature"))
    out.update(time_gradient_features(t, q, f"{phase}_capacity"))

    out.update(segment_cc_cv_features(t, v, i, phase))

    power = v * i

    out[f"{phase}_duration_s"] = safe_float(t[-1] - t[0])
    out[f"{phase}_n_points"] = int(t.numel())
    out[f"{phase}_phase_capacity"] = safe_float(q[-1])
    out[f"{phase}_energy_wh"] = safe_float(torch.trapz(power, t) / 3600.0)
    out[f"{phase}_abs_energy_wh"] = safe_float(torch.trapz(torch.abs(power), t) / 3600.0)
    out[f"{phase}_power_mean_w"] = safe_float(power.mean())
    out[f"{phase}_power_max_w"] = safe_float(power.max())
    out[f"{phase}_power_min_w"] = safe_float(power.min())

    if prev_capacity is None or not np.isfinite(prev_capacity):
        out[f"{phase}_capacity_diff_prev1"] = np.nan
        out[f"{phase}_capacity_ratio_prev1"] = np.nan
    else:
        cap_now = safe_float(q[-1])
        out[f"{phase}_capacity_diff_prev1"] = cap_now - prev_capacity
        out[f"{phase}_capacity_ratio_prev1"] = cap_now / prev_capacity if prev_capacity != 0 else np.nan

    cycle_life = safe_float(cycle.get("cycle_life"))
    cycle_number = safe_float(cycle.get("cycle_number"))
    initial_capacity = safe_float(cycle.get("initial_capacity_ah"))

    remaining_cycles = (
        cycle_life - cycle_number
        if np.isfinite(cycle_life) and np.isfinite(cycle_number)
        else np.nan 
    )

    out["remaining_cycles"] = remaining_cycles
    out["remaining_cycles_norm"] = (
        remaining_cycles / cycle_life
        if np.isfinite(cycle_life) and cycle_life > 0
        else np.nan
    )
    out["soh"] = (
        safe_float(q[-1]) / initial_capacity
        if np.isfinite(initial_capacity) and initial_capacity > 0
        else np.nan
    )

    out["dataset"] = cycle.get("dataset", "")
    out["source_file"] = cycle.get("source_file", "")
    out["battery_id"] = cycle.get("battery_id", "")
    out["cycle_index"] = cycle.get("cycle_index", np.nan)
    out["cycle_number"] = cycle.get("cycle_number", np.nan)
    out["cycle_life"] = cycle.get("cycle_life", np.nan)
    out["phase"] = phase

    return out, safe_float(q[-1])


def process_phase(phase):
    pattern = os.path.join(INTERIM_DIR, phase, f"*_{phase}_clean_cycles.parquet")
    files = sorted(glob(pattern))

    if not files:
        print(f"[WARN] Nenhum arquivo encontrado para fase: {phase}")
        return

    out_dir = os.path.join(PROCESSED_DIR, phase)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    output_file = os.path.join(out_dir, f"{phase}_features_gpu.parquet")
    log_file = os.path.join(REPORT_DIR, f"{phase}_features_gpu_log.json")

    all_features = []

    log = {
        "phase": phase,
        "device": str(DEVICE),
        "parquet_engine": PARQUET_ENGINE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "p_points": P_POINTS,
        "started_at": datetime.now().isoformat(),
        "files": [],
        "total_cycles": 0,
        "processed_cycles": 0,
        "errors": [],
    }

    print(f"\n[INFO] Phase: {phase}")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Parquet engine: {PARQUET_ENGINE}")
    print(f"[INFO] Files found: {len(files)}")

    for file_path in files:
        print(f"\n[INFO] Reading: {file_path}")

        df = pd.read_parquet(file_path, engine=PARQUET_ENGINE)

        input_rows_before_filter = int(len(df))

        if "is_model_usable" in df.columns:
            df = df[df["is_model_usable"] == True].copy()

        sort_cols = [c for c in ["battery_id", "cycle_number", "cycle_index"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, na_position="last")

        file_info = {
            "file": file_path,
            "input_rows_before_filter": input_rows_before_filter,
            "input_rows_after_filter": int(len(df)),
            "processed_rows": 0,
            "errors": 0,
        }

        prev_capacity_by_battery = {}

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {phase}"):
            try:
                battery_id = str(row.get("battery_id", ""))
                prev_capacity = prev_capacity_by_battery.get(battery_id)

                feats, cap_now = extract_one_cycle(row, phase, prev_capacity=prev_capacity)

                all_features.append(feats)
                prev_capacity_by_battery[battery_id] = cap_now

                log["processed_cycles"] += 1
                file_info["processed_rows"] += 1

            except Exception as e:
                file_info["errors"] += 1
                if len(log["errors"]) < 300:
                    log["errors"].append({
                        "file": file_path,
                        "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                        "error": str(e),
                    })

        log["total_cycles"] += int(len(df))
        log["files"].append(file_info)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_out = pd.DataFrame(all_features)

    df_out.to_parquet(output_file, index=False, engine=PARQUET_ENGINE)

    log["finished_at"] = datetime.now().isoformat()
    log["output_file"] = output_file
    log["output_shape"] = list(df_out.shape)
    log["columns"] = df_out.columns.tolist()

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved: {output_file}")
    print(f"[OK] Shape: {df_out.shape}")
    print(f"[OK] Log: {log_file}")


def main():
    print("[INFO] GPU feature extraction with fastparquet")
    print(f"[INFO] Torch: {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    for phase in PHASES:
        process_phase(phase)


if __name__ == "__main__":
    main()