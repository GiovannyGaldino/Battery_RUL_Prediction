#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATCH 03 - run_modeling_final_campaign_hpc.py (runner da campanha)

Todas as mudancas preservam o comportamento v2 por default.

  1. feature_group: o grupo stage_gap e testado ANTES de time/voltage/current/
     power/capacity/energy/temperature. Em v2 ele era testado por ultimo e ficava
     vazio, o que tornava no_stage_gap == all_features e
     global_stats_only == no_curves. Isso MUDA a composicao de duas das treze
     configuracoes; por isso ha --legacy-feature-groups para reproduzir v2.
  2. regression_metrics: alem das metricas brutas, grava MAE_clip0, RMSE_clip0,
     R2_clip0, nMAE_clip0, nRMSE_clip0, n_pred_negative e share_pred_negative,
     com y_pred truncado em zero. As colunas antigas nao mudam.
  3. --model-random-state: semente dos estimadores separada da do split
     (default: igual a --random-state, como em v2).
  4. --elasticnet-cv: troca ElasticNet(alpha=0.01) por ElasticNetCV ajustado
     dentro do treino (l1_ratio in {0.1,0.5,0.9}, trajetoria de 20 alphas gerada
     pelo sklearn com eps=1e-3, cv=5, max_iter=5000, n_jobs=model_n_jobs).
  5. --exclude-datasets: remove dominios de X/y/meta apos a carga. Viabiliza a
     selecao nested source-only (o alvo some de TODO treino da selecao).
  6. run_config.json passa a registrar tambem a versao do pacote (campanha_v3).
  7. Novo experimento no_causal_trend (todas as features menos causal_*): e o
     braco "tendencia removida", para comparar com all_features (causal) e com
     a campanha v2 (com vazamento).
"""
import sys, pathlib, py_compile
RAIZ = pathlib.Path(__file__).resolve().parent.parent
SRC = RAIZ / "pristine" / "run_modeling_final_campaign_hpc.py"
DST = RAIZ / "patched" / "run_modeling_final_campaign_hpc.py"
txt = SRC.read_text(encoding="utf-8"); log = []
def troca(velho, novo, nome):
    global txt
    n = txt.count(velho)
    if n != 1: sys.exit(f"ABORTADO [{nome}]: ancora ocorre {n}x\n{velho[:100]!r}")
    txt = txt.replace(velho, novo, 1); log.append(f"  ok  {nome}")

# 1 ---------------------------------------------------------------- feature_group
troca('''def feature_group(col: str) -> str:
    c = col.lower()
    if c.startswith("qv_") or c.startswith("vq_") or c.startswith("eq_"):
        return "curves"
    if "qv_" in c or "vq_" in c or "eq_" in c or "curve" in c:
        return "curves"
    if "capacity" in c or "capacityh" in c or "soh" in c:
        return "capacity"''', '''LEGACY_FEATURE_GROUPS = False   # v3: definido por --legacy-feature-groups


def feature_group(col: str) -> str:
    c = col.lower()
    if c.startswith("qv_") or c.startswith("vq_") or c.startswith("eq_"):
        return "curves"
    if "qv_" in c or "vq_" in c or "eq_" in c or "curve" in c:
        return "curves"
    # v3: stage_gap ANTES das familias de sinal. Em v2 este teste vinha por
    # ultimo e nunca casava (time_gap_* caia em time_sampling, voltage_gap_jump_n
    # em voltage, ...), deixando o grupo vazio.
    if not LEGACY_FEATURE_GROUPS and (
        "stage" in c or "_gap_" in c or c.endswith("_gap") or "rest" in c or "jump" in c or "transition" in c
    ):
        return "stage_gap"
    if "capacity" in c or "capacityh" in c or "soh" in c:
        return "capacity"''', "feature_group: stage_gap primeiro")

# 2 ---------------------------------------------------------------- metricas truncadas
troca('''    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if y_true.size >= 2 and np.unique(y_true).size >= 2 else np.nan
    smape = 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    scale = float(np.nanmean(np.abs(y_true))) + 1e-8

    return {''', '''    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if y_true.size >= 2 and np.unique(y_true).size >= 2 else np.nan
    smape = 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    scale = float(np.nanmean(np.abs(y_true))) + 1e-8

    # v3: mesmas metricas com a previsao truncada no dominio admissivel (RUL >= 0).
    # As colunas brutas acima sao mantidas intactas; estas sao adicionais.
    y_clip = np.maximum(y_pred, 0.0)
    n_neg = int((y_pred < 0).sum())
    mae_c = mean_absolute_error(y_true, y_clip)
    rmse_c = math.sqrt(mean_squared_error(y_true, y_clip))
    r2_c = r2_score(y_true, y_clip) if y_true.size >= 2 and np.unique(y_true).size >= 2 else np.nan

    return {
        "MAE_clip0": float(mae_c),
        "RMSE_clip0": float(rmse_c),
        "R2_clip0": float(r2_c) if np.isfinite(r2_c) else np.nan,
        "nMAE_clip0": float(mae_c / scale),
        "nRMSE_clip0": float(rmse_c / scale),
        "n_pred_negative": n_neg,
        "share_pred_negative": float(n_neg / y_true.size),''', "metricas com truncamento")
troca('''            "sMAPE": np.nan, "nMAE": np.nan, "nRMSE": np.nan,
            "y_mean": np.nan, "y_std": np.nan, "y_min": np.nan, "y_max": np.nan,
        }''', '''            "sMAPE": np.nan, "nMAE": np.nan, "nRMSE": np.nan,
            "y_mean": np.nan, "y_std": np.nan, "y_min": np.nan, "y_max": np.nan,
            "MAE_clip0": np.nan, "RMSE_clip0": np.nan, "R2_clip0": np.nan,
            "nMAE_clip0": np.nan, "nRMSE_clip0": np.nan,
            "n_pred_negative": 0, "share_pred_negative": np.nan,
        }''', "metricas vazias com as novas chaves")

# 3+4 -------------------------------------------------------------- modelos
troca('''def make_model_specs(random_state: int, model_n_jobs: int) -> Dict[str, Tuple[BaseEstimator, bool]]:
    """Return model_name -> (estimator, needs_scaling)."""
    return {
        "median": (MedianRegressor(), False),
        "elasticnet": (
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=50000, tol=1e-3, random_state=random_state, selection="random"),
            True,
        ),''', '''def make_model_specs(random_state: int, model_n_jobs: int, elasticnet_cv: bool = False) -> Dict[str, Tuple[BaseEstimator, bool]]:
    """Return model_name -> (estimator, needs_scaling).

    v3: `random_state` aqui e a semente dos ESTIMADORES (--model-random-state),
    que pode diferir da semente do split. `elasticnet_cv=True` substitui o
    alpha fixo por ElasticNetCV ajustado dentro do conjunto de treino.
    """
    if elasticnet_cv:
        from sklearn.linear_model import ElasticNetCV
        from sklearn import __version__ as _skv
        # Trajetoria de alphas gerada pelo sklearn a partir dos dados (de alpha_max
        # ate eps*alpha_max, 20 pontos, warm start ao longo da trajetoria). A grade
        # fixa logspace(-4,0) da v3.0.0 ficava inteira abaixo de alpha_max quando o
        # alvo esta em milhares de ciclos: quase sem regularizacao, sem convergir e
        # mais de uma hora por ajuste. `n_jobs` paraleliza folds x l1_ratio (15 tarefas).
        _kw = dict(l1_ratio=[0.1, 0.5, 0.9], eps=1e-3, cv=5, max_iter=5000, tol=1e-3,
                   random_state=random_state, selection="random", n_jobs=model_n_jobs)
        _major, _minor = (int(x) for x in _skv.split(".")[:2])
        if (_major, _minor) >= (1, 7):
            _kw["alphas"] = 20          # inteiro = numero de pontos da trajetoria
        else:
            _kw["n_alphas"] = 20        # parametro antigo (removido nas versoes recentes)
        enet = ElasticNetCV(**_kw)
    else:
        enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=50000, tol=1e-3, random_state=random_state, selection="random")
    return {
        "median": (MedianRegressor(), False),
        "elasticnet": (enet, True),''', "make_model_specs: semente e ElasticNetCV")

# 5+3+4+1 ---------------------------------------------------------- argparse
troca('''    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)''', '''    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42, help="Semente do SPLIT.")
    # ------------------------------------------------------------ v3
    parser.add_argument("--model-random-state", type=int, default=None,
                        help="Semente dos ESTIMADORES. Default: igual a --random-state (v2).")
    parser.add_argument("--elasticnet-cv", action="store_true",
                        help="ElasticNetCV ajustado dentro do treino em vez de alpha=0.01 fixo.")
    parser.add_argument("--exclude-datasets", nargs="*", default=None,
                        help="Dominios removidos de X/y/meta apos a carga (selecao nested source-only).")
    parser.add_argument("--legacy-feature-groups", action="store_true",
                        help="Reproduz o classificador de grupos de v2 (stage_gap vazio).")''', "argparse v3")

troca('''    args = parser.parse_args()
    args.input_root = args.input_root.expanduser()
    args.output_root = args.output_root.expanduser()
    args.experiments = expand_experiments(args.experiments)''', '''    args = parser.parse_args()
    args.input_root = args.input_root.expanduser()
    args.output_root = args.output_root.expanduser()
    args.experiments = expand_experiments(args.experiments)
    if args.model_random_state is None:
        args.model_random_state = args.random_state
    global LEGACY_FEATURE_GROUPS
    LEGACY_FEATURE_GROUPS = bool(args.legacy_feature_groups)''', "pos-processamento dos args")

# 7 ---------------------------------------------------------------- experimento no_causal_trend
troca('''    elif experiment == "no_stage_gap":
        selected = [c for c in cols if groups[c] != "stage_gap"]''', '''    elif experiment == "no_stage_gap":
        selected = [c for c in cols if groups[c] != "stage_gap"]
    elif experiment == "no_causal_trend":
        # v3: braco "removidas": sem as contrapartes causais de tendencia.
        # Comparar com all_features (causais) e com a campanha v2 (com vazamento).
        selected = [c for c in cols if not c.lower().startswith("causal_")]''', "experimento no_causal_trend")
troca('''        "resolution_p030",
    ]''', '''        "resolution_p030",
        "no_causal_trend",
    ]''', "no_causal_trend na lista 'all'")

# 6 ---------------------------------------------------------------- run_config
troca('''        serializable = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        json.dump(serializable, f, indent=2, ensure_ascii=False)''', '''        serializable = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        serializable["package"] = "campanha_v3"
        json.dump(serializable, f, indent=2, ensure_ascii=False)''', "run_config com versao")

troca('''    model_specs = make_model_specs(args.random_state, args.model_n_jobs)''',
      '''    model_specs = make_model_specs(args.model_random_state, args.model_n_jobs, elasticnet_cv=args.elasticnet_cv)''',
      "modelos com semente propria")

# 5 ---------------------------------------------------------------- exclude-datasets
troca('''        X, y, meta = load_phase(args.input_root, phase, args.target)
''', '''        X, y, meta = load_phase(args.input_root, phase, args.target)

        # v3: remove dominios inteiros (selecao nested: o alvo nao entra em nenhum treino)
        if args.exclude_datasets:
            keep = ~meta["dataset"].astype(str).isin([str(d) for d in args.exclude_datasets])
            n0 = len(X)
            X = X.loc[keep].reset_index(drop=True); y = y.loc[keep].reset_index(drop=True)
            meta = meta.loc[keep].reset_index(drop=True)
            print(f"[EXCLUDE] {phase}: removidos {args.exclude_datasets} -> {n0:,} -> {len(X):,} linhas")
''', "exclude-datasets apos a carga")

DST.write_text(txt, encoding="utf-8"); py_compile.compile(str(DST), doraise=True)
print("\n".join(log)); print(f"\ncompilado: {DST.name}")
