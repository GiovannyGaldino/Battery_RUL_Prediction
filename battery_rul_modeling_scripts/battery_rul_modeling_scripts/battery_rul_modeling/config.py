from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelingConfig:
    input_root: Path
    output_root: Path
    target: str = "target_rul80_steps_clipped"
    phases: List[str] = field(default_factory=lambda: ["charge", "discharge"])
    models: List[str] = field(default_factory=lambda: [
        "median", "elasticnet", "random_forest", "extra_trees", "hist_gradient_boosting", "mlp"
    ])
    experiments: List[str] = field(default_factory=lambda: ["all_features"])
    random_seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.15
    cv_folds: int = 3
    n_iter: int = 12
    n_jobs: int = -1
    max_missing_frac: float = 0.60
    min_variance: float = 1e-12
    mape_min_true: float = 1.0
    drop_zero_rul: bool = False
    quick: bool = False
    dataset_balanced_weights: bool = False
    max_train_rows: Optional[int] = None
    max_test_rows: Optional[int] = None
    allow_cycle_number_as_feature: bool = False
