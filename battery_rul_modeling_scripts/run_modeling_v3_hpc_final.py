
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from battery_rul_modeling.io_utils import load_phase_xy
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed

# Caminhos
features_root = Path("~/outputs/batteries_modeling_v3/modeling_ready").expanduser()
results_root = Path("~/outputs/batteries_modeling_v3/modeling_results").expanduser()
os.makedirs(results_root, exist_ok=True)

phases = ["charge", "discharge"]
models = {
    "ElasticNet": ElasticNet(max_iter=5000),
    "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200),
    "MLP": MLPRegressor(hidden_layer_sizes=(100,100), max_iter=500)
}

def load_phase_with_meta(phase):
    x_path = features_root / f"{phase}_X.parquet"
    y_path = features_root / f"{phase}_y.parquet"
    id_path = features_root / f"{phase}_id.parquet"
    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)
    meta = pd.read_parquet(id_path)
    if "battery_id" not in meta.columns:
        raise ValueError(f"'battery_id' não encontrado em {id_path}")
    return X, y, meta

def split_by_battery(X, y, meta, test_size=0.2, random_state=42):
    batteries = meta["battery_id"].unique()
    shuffled = pd.Series(batteries).sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_test = max(1, int(len(batteries)*test_size))
    test_batt = shuffled.iloc[:n_test]
    train_batt = shuffled.iloc[n_test:]
    train_idx = meta["battery_id"].isin(train_batt).values
    test_idx = meta["battery_id"].isin(test_batt).values
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], meta.iloc[train_idx], meta.iloc[test_idx]

def compute_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # Corrige y para array 1D
    y_true = y_true.values.ravel() if hasattr(y_true, "values") else np.ravel(y_true)
    y_pred = y_pred.ravel() if hasattr(y_pred, "ravel") else np.ravel(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    smape = 100*np.mean(np.abs(y_pred - y_true)/((np.abs(y_pred)+np.abs(y_true))/2 + 1e-8))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "sMAPE": smape}

def train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test, phase_dir):
    if name in ["ElasticNet", "MLP"]:
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    model_dir = phase_dir / name
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")
    pd.DataFrame([metrics]).to_csv(model_dir / "metrics.csv", index=False)
    pd.DataFrame({"y_true": y_test.values.ravel(), "y_pred": y_pred}).to_csv(model_dir / "predictions.csv", index=False)
    return metrics

# Pipeline principal
for phase in phases:
    print(f"=== Phase: {phase} ===")
    X, y, meta = load_phase_with_meta(phase)
    X_train, X_test, y_train, y_test, meta_train, meta_test = split_by_battery(X, y, meta)
    
    # Padronização para ElasticNet e MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    phase_dir = results_root / phase
    os.makedirs(phase_dir, exist_ok=True)
    
    # Paralelizar treino de modelos
    Parallel(n_jobs=len(models))(
        delayed(train_and_evaluate_model)(name, model, X_train_scaled if name in ["ElasticNet","MLP"] else X_train,
                                          y_train, X_test_scaled if name in ["ElasticNet","MLP"] else X_test,
                                          y_test, phase_dir)
        for name, model in models.items()
    )
