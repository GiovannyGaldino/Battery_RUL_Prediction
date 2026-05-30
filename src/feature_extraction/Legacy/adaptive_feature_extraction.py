import os
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import skew, kurtosis
import json
from tqdm import tqdm
from glob import glob

# -----------------------------
# Configurações de paths
# -----------------------------
ROOT_DIR = r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries"
INTERIM_DIR = os.path.join(ROOT_DIR, "data/interim")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data/processed")
LOG_DIR = os.path.join(ROOT_DIR, "data/reports/feature_extraction")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# Função de extração de features completa
# -----------------------------
def extract_features_full(cycle, phase="discharge", p=15, prev_cycle=None):
    features = {}
    time_s = np.array(cycle["time_s"])
    voltage_v = np.array(cycle["voltage_v"])
    capacity_ah = np.array(cycle["capacity_ah"])
    current_a = np.array(cycle.get("current_a", np.full_like(time_s, np.nan)))
    temperature_c = np.array(cycle.get("temperature_c", np.full_like(time_s, np.nan)))

    # Vetores amostrados Q(V)
    try:
        pchip = PchipInterpolator(voltage_v, capacity_ah)
        voltage_uniform = np.linspace(voltage_v.min(), voltage_v.max(), p)
        q_points = pchip(voltage_uniform)
        for i, val in enumerate(q_points, 1):
            features[f"{phase[:2].upper()}_{i}"] = val
        features[f"{phase[:2].upper()}_area"] = np.trapz(q_points, voltage_uniform)
        slopes = np.gradient(q_points, voltage_uniform)
        features[f"{phase[:2].upper()}_slope_mean"] = np.mean(slopes)
        features[f"{phase[:2].upper()}_slope_std"] = np.std(slopes)
        features[f"{phase[:2].upper()}_range"] = np.ptp(q_points)
        features[f"{phase[:2].upper()}_distance_to_initial"] = np.linalg.norm(q_points - q_points[0])
        features[f"{phase[:2].upper()}_max_dQdV"] = np.max(slopes)
        features[f"{phase[:2].upper()}_min_dQdV"] = np.min(slopes)
    except Exception:
        for i in range(1, p+1):
            features[f"{phase[:2].upper()}_{i}"] = np.nan
        features[f"{phase[:2].upper()}_area"] = np.nan
        features[f"{phase[:2].upper()}_slope_mean"] = np.nan
        features[f"{phase[:2].upper()}_slope_std"] = np.nan
        features[f"{phase[:2].upper()}_range"] = np.nan
        features[f"{phase[:2].upper()}_distance_to_initial"] = np.nan
        features[f"{phase[:2].upper()}_max_dQdV"] = np.nan
        features[f"{phase[:2].upper()}_min_dQdV"] = np.nan

    # Estatísticas de série
    for signal, name in zip([voltage_v, current_a, temperature_c], ["voltage", "current", "temperature"]):
        features[f"{phase}_{name}_mean"] = np.nanmean(signal)
        features[f"{phase}_{name}_min"] = np.nanmin(signal)
        features[f"{phase}_{name}_max"] = np.nanmax(signal)
        features[f"{phase}_{name}_std"] = np.nanstd(signal)
        features[f"{phase}_{name}_skew"] = skew(signal, nan_policy="omit")
        features[f"{phase}_{name}_kurtosis"] = kurtosis(signal, nan_policy="omit")
        for perc in [25,50,75]:
            features[f"{phase}_{name}_p{perc}"] = np.nanpercentile(signal, perc)

    # Energia
    features[f"{phase}_energy_wh"] = np.trapz(voltage_v*current_a, time_s)/3600
    # Capacidade final
    features[f"{phase}_phase_capacity"] = capacity_ah[-1]
    # Duração e pontos
    features[f"{phase}_duration_s"] = time_s[-1] - time_s[0]
    features[f"{phase}_n_points"] = len(time_s)
    # Gradiente de temperatura
    features[f"{phase}_dT_dt_mean"] = np.mean(np.gradient(temperature_c, time_s))

    # ΔQ em relação ao ciclo anterior
    if prev_cycle is not None:
        features[f"{phase}_capacity_diff_prev"] = capacity_ah[-1] - prev_cycle["capacity_ah"][-1]
    else:
        features[f"{phase}_capacity_diff_prev"] = np.nan

    # Variáveis alvo
    features["remaining_cycles"] = cycle["cycle_life"] - cycle["cycle_number"]
    features["remaining_cycles_norm"] = features["remaining_cycles"] / cycle["cycle_life"] if cycle["cycle_life"]>0 else np.nan
    features["soh"] = capacity_ah[-1] / cycle["initial_capacity_ah"] if cycle["initial_capacity_ah"]>0 else np.nan

    # Traceabilidade
    features["dataset"] = cycle.get("dataset","")
    features["battery_id"] = cycle.get("battery_id","")
    features["cycle_index"] = cycle.get("cycle_index",-1)
    features["cycle_number"] = cycle.get("cycle_number",-1)
    features["phase"] = phase

    return features

# -----------------------------
# Processa todos os arquivos de uma fase
# -----------------------------
def process_phase_full(phase):
    pattern = os.path.join(INTERIM_DIR, phase, f"*_{phase}_clean_cycles.parquet")
    files = glob(pattern)
    all_features = []
    log = {"files_processed": [], "total_cycles":0, "processed_cycles":0, "errors":[]}

    for file_path in files:
        df_raw = pd.read_parquet(file_path)
        log["files_processed"].append(file_path)
        log["total_cycles"] += len(df_raw)
        prev_cycle_data = None

        for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc=f"Processing {phase}"):
            try:
                feats = extract_features_full(row.to_dict(), phase=phase, prev_cycle=prev_cycle_data)
                all_features.append(feats)
                prev_cycle_data = row
                log["processed_cycles"] +=1
            except Exception as e:
                log["errors"].append({"file":file_path,"index":idx,"error":str(e)})

    # Salvar features
    output_file = os.path.join(PROCESSED_DIR, phase, f"{phase}_features_full.parquet")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(all_features).to_parquet(output_file,index=False)

    # Salvar log
    log_file = os.path.join(LOG_DIR, f"{phase}_full_extract_log.json")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file,"w") as f:
        json.dump(log, f, indent=4)

    print(f"{phase.capitalize()} features salvas: {len(all_features)} ciclos.")

# -----------------------------
# Execução principal
# -----------------------------
if __name__=="__main__":
    for phase in ["charge","discharge"]:
        process_phase_full(phase)