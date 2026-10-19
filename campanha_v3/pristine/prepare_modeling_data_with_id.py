#!/usr/bin/env python3
"""
prepare_modeling_data_with_id.py
Prepare battery datasets for RUL modeling with identification dataframe.
"""

import pandas as pd
from pathlib import Path
import argparse

_LEAKAGE_EXACT = {"cycle_life","remaining_cycles","remaining_cycles_norm",
    "eol_criterion","eol_capacity_ah","initial_capacity_ah","nominal_capacity_ah","soh"}
_PREFIXES = ("target_", "input_")
_SUBSTR = ("rul", "remaining_cycles", "eol", "after_eol")


def prepare_phase(df, target_column, exclude_cols, id_cols, phase_name="charge"):
    print(f"[INFO] Preparing phase: {phase_name}")
    
    # Filtrar apenas registros com RUL positivo
    df = df[df[target_column] > 0].copy()
    
    # Garantir que EOL foi detectado (is_valid_cycle ou similar)
    if "is_valid_cycle" in df.columns:
        df = df[df["is_valid_cycle"] == True].copy()
    
    # Separar features preditoras
    _drop = [c for c in exclude_cols if c in df.columns] + [target_column]
    for c in df.columns:
        if c in _drop: continue
        lc = str(c).lower()
        if c in _LEAKAGE_EXACT or str(c).startswith(_PREFIXES) or any(x in lc for x in _SUBSTR):
            _drop.append(c)
    _nrem = len([c for c in _drop if c in df.columns])
    features = df.drop(columns=[c for c in _drop if c in df.columns])
    features = features.select_dtypes(include=["number"])
    # NaN PRESERVADOS: a SimpleImputer do pipeline imputa dentro da dobra de treino.
    _nan = float(features.isna().to_numpy().mean()) if len(features) else 0.0
    print(f"  colunas removidas: {_nrem} | NaN preservados: {100*_nan:.2f}% das celulas")
    
    # Target
    target = df[target_column].to_frame()
    
    # Criar dataframe de identificação usando apenas colunas existentes
    existing_id_cols = [c for c in id_cols if c in df.columns]
    ids = df[existing_id_cols].reset_index(drop=True)
    
    print(f"  {phase_name}: {features.shape[0]} samples x {features.shape[1]} features")
    
    return features, target, ids

def main():
    parser = argparse.ArgumentParser(description="Prepare battery datasets for modeling with ID dataframe")
    parser.add_argument("--features-root", type=str, required=True,
                        help="Folder containing charge/discharge *_with_curves.parquet files")
    parser.add_argument("--output-root", type=str, required=True,
                        help="Folder to save prepared X/y/id files")
    parser.add_argument("--target", type=str, default="target_rul80_steps_clipped",
                        help="Target column for RUL modeling")
    args = parser.parse_args()
    
    features_root = Path(args.features_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Colunas a remover (leakage e identificação)
    exclude_cols = [
        "dataset", "battery_id", "source_file", "source_path", "cycle_number",
        "cycle_index", "phase", "cycle_life", "remaining_cycles", "eol_capacity_ah",
        "target_rul70_steps_clipped", "target_initial_capacity_ah", "target_soh_recomputed"
    ]
    
    # Colunas de identificação possíveis
    id_cols = ["dataset", "battery_id", "source_file", "cycle_number", "cycle_index", "phase"]
    
    for phase_file, phase_name in [("charge_phase_modeling_with_curves.parquet", "charge"),
                                   ("discharge_phase_modeling_with_curves.parquet", "discharge")]:
        file_path = features_root / phase_file
        if not file_path.exists():
            print(f"[WARNING] File not found: {file_path}")
            continue
        
        print(f"[INFO] Loading {phase_name} data from {file_path}...")
        df = pd.read_parquet(file_path)
        
        X, y, ids = prepare_phase(df, args.target, exclude_cols, id_cols, phase_name=phase_name)
        
        # Salvar arquivos finais
        X_file = output_root / f"{phase_name}_X.parquet"
        y_file = output_root / f"{phase_name}_y.parquet"
        id_file = output_root / f"{phase_name}_id.parquet"
        
        X.to_parquet(X_file)
        y.to_parquet(y_file)
        ids.to_parquet(id_file)
        
        print(f"[INFO] Saved {phase_name} features to {X_file}")
        print(f"[INFO] Saved {phase_name} target to {y_file}")
        print(f"[INFO] Saved {phase_name} identification to {id_file}")
    
    print("[INFO] Data preparation complete. Ready for model training.")

if __name__ == "__main__":
    main()
