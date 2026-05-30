import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------
# Configurações do pipeline
# -------------------------------
resolutions = [3, 15, 30]
base_dir = r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\interim"
output_dir = r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\processed"
os.makedirs(output_dir, exist_ok=True)
batch_size = 50  # novo tamanho do bloco

# -------------------------------
# Funções auxiliares
# -------------------------------
def integrate_capacity(current, time):
    current = np.array(current)
    time = np.array(time)
    cap = np.cumsum((np.abs(current[1:] + current[:-1]) / 2) * (time[1:] - time[:-1]) / 3600.0)
    return np.concatenate([[0.0], cap])

def compute_QV_features(voltage, capacity, p):
    voltage = np.array(voltage)
    capacity = np.array(capacity)
    valid = np.isfinite(voltage) & np.isfinite(capacity)
    voltage, capacity = voltage[valid], capacity[valid]
    if len(voltage) < 2:
        return np.full(p, np.nan), {}
    df = pd.DataFrame({'v': voltage, 'q': capacity})
    df = df.groupby('v', as_index=False).mean()
    voltage_sorted, capacity_sorted = df['v'].values, df['q'].values
    if len(voltage_sorted) < 2:
        return np.full(p, np.nan), {}
    k = min(3, len(voltage_sorted)-1)
    if k < 1:
        return np.full(p, np.nan), {}
    spline = UnivariateSpline(voltage_sorted, capacity_sorted, s=1e-5, k=k)
    V_samples = np.linspace(voltage_sorted.min(), voltage_sorted.max(), p)
    QV_points = spline(V_samples)
    slope = np.diff(QV_points)/np.diff(V_samples)
    return QV_points, {
        "QV_area": np.sum((QV_points[1:] + QV_points[:-1]) / 2 * np.diff(V_samples)),
        "QV_range": QV_points.max() - QV_points.min(),
        "QV_slope_mean": slope.mean(),
        "QV_slope_std": slope.std(),
        "QV_slope_max": np.max(np.abs(slope))
    }

def compute_VQ_features(voltage, capacity, p):
    voltage = np.array(voltage)
    capacity = np.array(capacity)
    valid = np.isfinite(voltage) & np.isfinite(capacity)
    voltage, capacity = voltage[valid], capacity[valid]
    if len(voltage) < 2:
        return np.full(p, np.nan), {}
    df = pd.DataFrame({'v': voltage, 'q': capacity})
    df = df.groupby('q', as_index=False).mean()
    capacity_sorted, voltage_sorted = df['q'].values, df['v'].values
    if len(capacity_sorted) < 2:
        return np.full(p, np.nan), {}
    k = min(3, len(capacity_sorted)-1)
    if k < 1:
        return np.full(p, np.nan), {}
    spline = UnivariateSpline(capacity_sorted, voltage_sorted, s=1e-5, k=k)
    Q_samples = np.linspace(capacity_sorted.min(), capacity_sorted.max(), p)
    VQ_points = spline(Q_samples)
    slope = np.diff(VQ_points)/np.diff(Q_samples)
    return VQ_points, {
        "VQ_area": np.sum((VQ_points[1:] + VQ_points[:-1]) / 2 * np.diff(Q_samples)),
        "VQ_range": VQ_points.max() - VQ_points.min(),
        "VQ_slope_mean": slope.mean(),
        "VQ_slope_std": slope.std(),
        "VQ_slope_max": np.max(np.abs(slope))
    }

def euclidean_distance(arr1, arr2):
    arr1, arr2 = np.array(arr1), np.array(arr2)
    valid = np.isfinite(arr1) & np.isfinite(arr2)
    if valid.sum() == 0:
        return np.nan
    return np.sqrt(np.sum((arr1[valid]-arr2[valid])**2)/valid.sum())

# -------------------------------
# Extração de features de um ciclo
# -------------------------------
def extract_cycle_features(cycle, prev1=None, prev5=None, initial=None, p_values=resolutions):
    feats = {}
    voltage = np.array(cycle['voltage_v'])
    current = np.array(cycle['current_a'])
    temp = np.array(cycle.get('temperature_c', [np.nan]*len(voltage)))
    capacity = np.array(cycle.get('capacity_ah', integrate_capacity(current, cycle['time_s'])))

    feats['phase_capacity_ah'] = capacity[-1] if len(capacity) > 0 else np.nan
    feats['T_min'] = np.nanmin(temp)
    feats['T_max'] = np.nanmax(temp)
    feats['T_mean'] = np.nanmean(temp)
    feats['T_skew'] = skew(temp, nan_policy='omit') if np.nanstd(temp) > 1e-6 else 0
    feats['T_kurt'] = kurtosis(temp, nan_policy='omit') if np.nanstd(temp) > 1e-6 else 0
    feats['n_points'] = len(voltage)
    feats['duration_s'] = cycle['time_s'][-1] - cycle['time_s'][0] if len(cycle['time_s'])>1 else 0

    QV_dict, VQ_dict = {}, {}
    for p in p_values:
        QV_pts, QV_stats = compute_QV_features(voltage, capacity, p)
        VQ_pts, VQ_stats = compute_VQ_features(voltage, capacity, p)
        QV_dict[p] = QV_pts
        VQ_dict[p] = VQ_pts
        for i in range(p):
            feats[f'QV_{p}_{i+1}'] = QV_pts[i]
            feats[f'VQ_{p}_{i+1}'] = VQ_pts[i]
        feats.update({f'{k}_p{p}': v for k,v in QV_stats.items()})
        feats.update({f'{k}_p{p}': v for k,v in VQ_stats.items()})

    for p in p_values:
        feats[f'QV_dist_initial_p{p}'] = euclidean_distance(QV_dict[p], initial[f'QV_{p}']) if initial and f'QV_{p}' in initial else np.nan
        feats[f'VQ_dist_initial_p{p}'] = euclidean_distance(VQ_dict[p], initial[f'VQ_{p}']) if initial and f'VQ_{p}' in initial else np.nan
        feats[f'QV_dist_prev1_p{p}'] = euclidean_distance(QV_dict[p], prev1[f'QV_{p}']) if prev1 and f'QV_{p}' in prev1 else np.nan
        feats[f'VQ_dist_prev1_p{p}'] = euclidean_distance(VQ_dict[p], prev1[f'VQ_{p}']) if prev1 and f'VQ_{p}' in prev1 else np.nan
        feats[f'QV_dist_prev5_p{p}'] = euclidean_distance(QV_dict[p], prev5[f'QV_{p}']) if prev5 and f'QV_{p}' in prev5 else np.nan
        feats[f'VQ_dist_prev5_p{p}'] = euclidean_distance(VQ_dict[p], prev5[f'VQ_{p}']) if prev5 and f'VQ_{p}' in prev5 else np.nan

    feats.update({
        'cycle_number': cycle['cycle_number'],
        'cycle_index': cycle['cycle_index'],
        'cycle_life': cycle['cycle_life'],
        'remaining_cycles': cycle['cycle_life']-cycle['cycle_number'],
        'phase': cycle['phase'],
        'dataset': cycle['dataset'],
        'battery_id': cycle['battery_id'],
        'soh': cycle.get('soh', np.nan)
    })
    return feats

# -------------------------------
# Processa um bloco de ciclos
# -------------------------------
def process_block(cycles_block, features_cumulativas):
    feats_block = []
    for i, cycle in enumerate(cycles_block):
        prev1 = features_cumulativas[-1] if len(features_cumulativas) >= 1 else None
        prev5 = features_cumulativas[-5] if len(features_cumulativas) >= 5 else None
        initial = features_cumulativas[0] if len(features_cumulativas) >= 1 else None
        feats_block.append(extract_cycle_features(cycle, prev1=prev1, prev5=prev5, initial=initial))
    return feats_block

# -------------------------------
# Processa arquivo em blocos com log do tamanho real
# -------------------------------
def process_file_in_batches(file_path, batch_size=50):
    df_raw = pd.read_parquet(file_path)
    group_cols = ['dataset','battery_id','phase']

    for _, group in df_raw.groupby(group_cols):
        group_sorted = group.sort_values('cycle_number')
        cycles_list = group_sorted.to_dict(orient='records')
        output_file = os.path.join(output_dir, f"{group.iloc[0]['phase']}_{group.iloc[0]['battery_id']}_features.parquet")
        features_cumulativas = []

        total_blocks = (len(cycles_list)+batch_size-1)//batch_size
        with tqdm(total=total_blocks, desc=f"{group.iloc[0]['battery_id']}") as pbar:
            for start in range(0, len(cycles_list), batch_size):
                block = cycles_list[start:start+batch_size]
                print(f"Processando bloco {start} a {start+len(block)-1} ({len(block)} ciclos)")
                feats_block = process_block(block, features_cumulativas)
                features_cumulativas.extend(feats_block)

                df_block = pd.DataFrame(feats_block)
                if not os.path.exists(output_file):
                    df_block.to_parquet(output_file, index=False)
                else:
                    df_block.to_parquet(output_file, index=False, engine='fastparquet', append=True)

                pbar.update(1)
                del feats_block, df_block

# -------------------------------
# Main
# -------------------------------
def main():
    for phase in ['charge','discharge']:
        folder_path = os.path.join(base_dir, phase)
        files = glob.glob(os.path.join(folder_path, "*.parquet"))
        print(f"Found {len(files)} {phase} files")
        from multiprocessing import freeze_support
        freeze_support()
        for file_path in files:
            process_file_in_batches(file_path, batch_size=batch_size)

if __name__=="__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()