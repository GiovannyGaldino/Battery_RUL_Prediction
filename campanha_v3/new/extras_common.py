"""Carga do runner e da matriz para os scripts extras (importancia, curvas de aprendizado)."""
import importlib.util, sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def load_runner(path: Path | None = None):
    path = path or RAIZ / "patched" / "run_modeling_final_campaign_hpc.py"
    spec = importlib.util.spec_from_file_location("runner", path)
    r = importlib.util.module_from_spec(spec); sys.modules["runner"] = r; spec.loader.exec_module(r)
    return r


def load_phase_experiment(r, input_root: Path, phase: str, target: str, experiment: str):
    X, y, meta = r.load_phase(input_root, phase, target)
    cols = r.select_experiment_features(X.columns, experiment)
    return X[cols], y, meta
