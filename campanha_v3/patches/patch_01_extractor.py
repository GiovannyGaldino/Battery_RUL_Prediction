#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATCH 01 - build_hierarchical_features_v3.py (construcao do alvo)

Aplica sobre pristine/ e escreve em patched/. Cada ancora tem de ocorrer
EXATAMENTE uma vez no arquivo original; caso contrario o patch aborta.

Mudancas (todas com default que reproduz o comportamento v2):
  1. find_eol_position devolve (pos, cycle, cap, reject_reason, min_ratio).
     Motivos: found | no_initial_capacity | never_crossed | no_persistence |
     permanence_violated | empty_series.
  2. Permanencia parametrizada: --eol-permanence-mode {absolute,margin},
     --eol-permanence-ratio (0.90), --eol-permanence-margin (0.10),
     --eol-permanence-on {raw,smooth}. Default = absolute 0.90 sobre raw (v2).
  3. Limiares por CLI: --eol-threshold (0.80) e --eol-secondary-threshold (0.70).
     Os nomes de coluna derivam do limiar: eol80/rul80, eol85/rul85, ...
  4. Novas colunas por limiar: target_eolNN_reject_reason,
     target_eolNN_min_ratio_reached, target_eolNN_threshold_ratio,
     target_eolNN_permanence_ratio. Idem no capacity_life_summary.
  5. build_base_record copia toda coluna target_* + passthrough, sem lista fixa
     de nomes (fica agnostico ao limiar).
  6. Flag morta --eol-allow-single-point-fallback removida, junto com o bloco
     inalcancavel.
"""
import sys, pathlib, re, py_compile

RAIZ = pathlib.Path(__file__).resolve().parent.parent
SRC = RAIZ / "pristine" / "build_hierarchical_features_v3.py"
DST = RAIZ / "patched" / "build_hierarchical_features_v3.py"
txt = SRC.read_text(encoding="utf-8")
log = []

def uma(ancora, nome):
    n = txt.count(ancora)
    if n != 1:
        sys.exit(f"ABORTADO [{nome}]: ancora ocorre {n}x, esperado 1:\n{ancora[:120]!r}")

def troca(velho, novo, nome):
    global txt
    uma(velho, nome)
    txt = txt.replace(velho, novo, 1)
    log.append(f"  ok  {nome}")

# ---------------------------------------------------------------- 1+2: find_eol_position
ini = "def find_eol_position("
fim = "def compute_trend_stats("
uma(ini, "inicio find_eol_position"); uma(fim, "inicio compute_trend_stats")
a, b = txt.index(ini), txt.index(fim)
NOVA_FUNC = '''def find_eol_position(
    cap: pd.Series,
    initial_capacity: float,
    threshold_ratio: float,
    rolling_window: int,
    consecutive: int,
    min_position: int,
    permanence_ratio: float = EOL_PERMANENCE_RATIO,
    permanence_on_smooth: bool = False,
) -> Tuple[float, float, float, str, float]:
    """
    Returns:
        eol_position, eol_cycle_like_value, eol_capacity_value,
        reject_reason, min_ratio_reached

    eol_position is the row order within the battery/phase group.
    reject_reason is "found" when an EOL was accepted; otherwise it names the
    single rule that blocked acceptance, so that right-censoring can be audited.
    min_ratio_reached is min(raw capacity) / initial_capacity, regardless of
    whether an EOL was found.

    v3 (campanha_v3): the permanence rule is now a parameter. v2 behaviour is
    reproduced with permanence_ratio=0.90 and permanence_on_smooth=False.
    """

    if not np.isfinite(initial_capacity) or initial_capacity <= 0:
        return np.nan, np.nan, np.nan, "no_initial_capacity", np.nan

    values = pd.to_numeric(cap, errors="coerce").astype(float)
    if len(values) == 0 or not values.notna().any():
        return np.nan, np.nan, np.nan, "empty_series", np.nan

    vals_np = values.to_numpy(dtype=float)
    min_ratio = float(np.nanmin(vals_np) / initial_capacity)

    smooth = values.rolling(window=rolling_window, min_periods=1).median()
    threshold = threshold_ratio * initial_capacity
    below = (smooth <= threshold) & values.notna()

    consecutive = max(1, int(consecutive))
    min_position = max(0, int(min_position))
    arr = below.to_numpy(dtype=bool)

    if not bool(arr[min_position:].any()):
        return np.nan, np.nan, np.nan, "never_crossed", min_ratio

    # O fim de vida e irreversivel. Um cruzamento so e aceito se, a partir dele,
    # a capacidade nunca mais superar permanence_ratio da capacidade inicial.
    # Em v2 essa checagem era feita sobre valores brutos (um unico outlier
    # tardio anulava o cruzamento). Em v3 pode ser feita sobre a serie suavizada.
    permanence = permanence_ratio * initial_capacity
    perm_series = smooth.to_numpy(dtype=float) if permanence_on_smooth else vals_np

    found_pos: Optional[int] = None
    saw_candidate = False
    start = int(min_position)
    while start < len(arr):
        cand: Optional[int] = None
        for pos in range(start, len(arr)):
            end = min(len(arr), pos + consecutive)
            if end - pos < consecutive:
                break
            if bool(arr[pos:end].all()):
                cand = pos
                break
        if cand is None:
            break
        saw_candidate = True
        posterior = perm_series[cand:]
        viola = np.where(np.isfinite(posterior) & (posterior > permanence))[0]
        if len(viola) == 0:
            found_pos = cand
            break
        start = cand + int(viola[-1]) + 1

    if found_pos is None:
        reason = "permanence_violated" if saw_candidate else "no_persistence"
        return np.nan, np.nan, np.nan, reason, min_ratio

    eol_capacity = safe_float(values.iloc[found_pos])
    return float(found_pos), np.nan, eol_capacity, "found", min_ratio


'''
txt = txt[:a] + NOVA_FUNC + txt[b:]
log.append("  ok  find_eol_position reescrita (motivo de rejeicao + permanencia parametrizada)")

# ---------------------------------------------------------------- 5: build_base_record agnostico
troca('''    # Add rebuilt targets.
    for col in TARGET_OUTPUT_COLUMNS:
        if col in target_row:
            base[col] = target_row[col]
''', '''    # Add rebuilt targets. v3: copia toda coluna target_* mais as de passagem,
    # sem depender de uma lista fixa de nomes (os nomes derivam do limiar).
    for col, val in target_row.items():
        if col.startswith("target_") or col in TARGET_PASSTHROUGH_COLUMNS:
            base[col] = val
''', "build_base_record agnostico ao limiar")

troca('''TARGET_OUTPUT_COLUMNS = [
    "source_file_key",''', '''# v3: colunas que saem de build_targets e NAO sao alvo, mas devem acompanhar o
# registro. Tudo que comeca com target_ e copiado automaticamente.
TARGET_PASSTHROUGH_COLUMNS = {
    "source_file_key", "source_row_id", "phase_folder", "dataset", "battery_id",
    "cycle_index", "cycle_number", "phase",
    "capacity_trend_slope_per_step", "capacity_trend_spearman",
    "capacity_positive_step_rate", "capacity_negative_step_rate",
    "capacity_median_abs_step_ah",
    "capacity_validation_status", "capacity_validation_notes",
    "input_soh", "input_remaining_cycles", "input_remaining_cycles_norm",
    "input_cycle_life", "input_initial_capacity_ah", "input_nominal_capacity_ah",
}

# Mantida por compatibilidade com scripts externos (nomes do limiar 80/70).
TARGET_OUTPUT_COLUMNS = [
    "source_file_key",''', "TARGET_PASSTHROUGH_COLUMNS")

# ---------------------------------------------------------------- 3+4: build_targets
troca('''def build_targets(
    scalar_df: pd.DataFrame,
    initial_method: str,
    initial_window: int,
    initial_fraction: float,
    eol_rolling_window: int,
    eol_consecutive: int,
    eol_min_position: int,
    drop_post_eol: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:''', '''def build_targets(
    scalar_df: pd.DataFrame,
    initial_method: str,
    initial_window: int,
    initial_fraction: float,
    eol_rolling_window: int,
    eol_consecutive: int,
    eol_min_position: int,
    drop_post_eol: bool = True,
    eol_threshold: float = 0.80,
    eol_secondary_threshold: float = 0.70,
    permanence_mode: str = "absolute",
    permanence_ratio: float = EOL_PERMANENCE_RATIO,
    permanence_margin: float = 0.10,
    permanence_on_smooth: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:''', "assinatura build_targets")

troca('''    cycle_sort_col = "cycle_index" if "cycle_index" in scalar_df.columns else "cycle_number"

    records: List[pd.DataFrame] = []''', '''    cycle_sort_col = "cycle_index" if "cycle_index" in scalar_df.columns else "cycle_number"

    # v3: nomes de coluna derivam do limiar (0.80 -> eol80/rul80, 0.85 -> eol85/rul85).
    P = int(round(eol_threshold * 100))
    S = int(round(eol_secondary_threshold * 100))
    if P == S:
        raise ValueError("eol_threshold e eol_secondary_threshold nao podem ser iguais")
    if permanence_mode == "margin":
        perm_p = float(eol_threshold + permanence_margin)
        perm_s = float(eol_secondary_threshold + permanence_margin)
    elif permanence_mode == "absolute":
        perm_p = perm_s = float(permanence_ratio)
    else:
        raise ValueError(f"permanence_mode invalido: {permanence_mode!r}")

    records: List[pd.DataFrame] = []''', "derivacao de nomes e permanencia")

troca('''        eol80_pos, _, eol80_capacity = find_eol_position(
            cap=cap_ref,
            initial_capacity=initial_capacity,
            threshold_ratio=0.80,
            rolling_window=eol_rolling_window,
            consecutive=eol_consecutive,
            min_position=eol_min_position,
        )

        eol70_pos, _, eol70_capacity = find_eol_position(
            cap=cap_ref,
            initial_capacity=initial_capacity,
            threshold_ratio=0.70,
            rolling_window=eol_rolling_window,
            consecutive=eol_consecutive,
            min_position=eol_min_position,
        )''', '''        eol80_pos, _, eol80_capacity, eol80_reason, eol80_min_ratio = find_eol_position(
            cap=cap_ref,
            initial_capacity=initial_capacity,
            threshold_ratio=eol_threshold,
            rolling_window=eol_rolling_window,
            consecutive=eol_consecutive,
            min_position=eol_min_position,
            permanence_ratio=perm_p,
            permanence_on_smooth=permanence_on_smooth,
        )

        eol70_pos, _, eol70_capacity, eol70_reason, eol70_min_ratio = find_eol_position(
            cap=cap_ref,
            initial_capacity=initial_capacity,
            threshold_ratio=eol_secondary_threshold,
            rolling_window=eol_rolling_window,
            consecutive=eol_consecutive,
            min_position=eol_min_position,
            permanence_ratio=perm_s,
            permanence_on_smooth=permanence_on_smooth,
        )''', "chamadas de find_eol_position")

# renomeia as chaves literais SOMENTE dentro de build_targets
ini = "def build_targets("; fim = "    targets = pd.concat(records, ignore_index=True, sort=False) if records else pd.DataFrame()"
uma(ini, "inicio build_targets"); uma(fim, "fim de build_targets")
a, b = txt.index(ini), txt.index(fim)
corpo = txt[a:b]
n_antes = corpo.count('"target_eol80_') + corpo.count('"target_eol70_') + corpo.count('"target_rul80_') + corpo.count('"target_rul70_') + corpo.count('"target_after_eol80"') + corpo.count('"target_after_eol70"')
for velho, novo in [
    ('"target_eol80_', 'f"target_eol{P}_'), ('"target_eol70_', 'f"target_eol{S}_'),
    ('"target_rul80_', 'f"target_rul{P}_'), ('"target_rul70_', 'f"target_rul{S}_'),
    ('"target_after_eol80"', 'f"target_after_eol{P}"'), ('"target_after_eol70"', 'f"target_after_eol{S}"'),
    ('"eol80_found"', 'f"eol{P}_found"'), ('"eol80_order_index"', 'f"eol{P}_order_index"'),
    ('"eol80_cycle_index"', 'f"eol{P}_cycle_index"'), ('"eol80_capacity_ah"', 'f"eol{P}_capacity_ah"'),
    ('"eol70_found"', 'f"eol{S}_found"'), ('"eol70_order_index"', 'f"eol{S}_order_index"'),
    ('"eol70_cycle_index"', 'f"eol{S}_cycle_index"'), ('"eol70_capacity_ah"', 'f"eol{S}_capacity_ah"'),
]:
    corpo = corpo.replace(velho, novo)
txt = txt[:a] + corpo + txt[b:]
log.append(f"  ok  {n_antes} chaves literais de coluna renomeadas para f-strings dentro de build_targets")

# novas colunas no target_part (ancora: a linha do capacity_validation_status no dict)
troca('''                "capacity_validation_status": status,
                "capacity_validation_notes": notes,
                "input_soh":''', '''                # v3: auditoria da censura
                f"target_eol{P}_reject_reason": eol80_reason,
                f"target_eol{P}_min_ratio_reached": eol80_min_ratio,
                f"target_eol{P}_threshold_ratio": float(eol_threshold),
                f"target_eol{P}_permanence_ratio": perm_p,
                f"target_eol{S}_reject_reason": eol70_reason,
                f"target_eol{S}_min_ratio_reached": eol70_min_ratio,
                f"target_eol{S}_threshold_ratio": float(eol_secondary_threshold),
                f"target_eol{S}_permanence_ratio": perm_s,
                "capacity_validation_status": status,
                "capacity_validation_notes": notes,
                "input_soh":''', "novas colunas no target_part")

# novas colunas no summary
troca('''                "trend_slope_per_step": trend["capacity_trend_slope_per_step"],''',
      '''                f"eol{P}_reject_reason": eol80_reason,
                f"eol{P}_min_ratio_reached": eol80_min_ratio,
                f"eol{P}_threshold_ratio": float(eol_threshold),
                f"eol{P}_permanence_ratio": perm_p,
                f"eol{S}_reject_reason": eol70_reason,
                f"eol{S}_min_ratio_reached": eol70_min_ratio,
                f"eol{S}_threshold_ratio": float(eol_secondary_threshold),
                f"eol{S}_permanence_ratio": perm_s,
                "trend_slope_per_step": trend["capacity_trend_slope_per_step"],''', "novas colunas no summary")

# ---------------------------------------------------------------- 6: argparse
troca('''    parser.add_argument("--keep-post-eol", action="store_true",
                        help="Mantem registros pos-EOL truncados em zero (comportamento antigo).")
    parser.add_argument("--eol-allow-single-point-fallback", action="store_true",
                        help="Reativa o fallback de ponto único (NAO recomendado: anula --eol-consecutive).")
''', '''    parser.add_argument("--keep-post-eol", action="store_true",
                        help="Mantem registros pos-EOL truncados em zero (comportamento antigo).")
    # v3: limiares e permanencia por linha de comando. Defaults reproduzem v2.
    parser.add_argument("--eol-threshold", type=float, default=0.80,
                        help="Limiar primario de fim de vida como fracao da capacidade inicial (v2: 0.80).")
    parser.add_argument("--eol-secondary-threshold", type=float, default=0.70,
                        help="Limiar secundario (v2: 0.70). Use 0.85 para o braco de sensibilidade a censura.")
    parser.add_argument("--eol-permanence-mode", choices=["absolute", "margin"], default="absolute",
                        help="absolute: permanencia fixa em --eol-permanence-ratio (v2). "
                             "margin: permanencia = limiar + --eol-permanence-margin, escala com o limiar.")
    parser.add_argument("--eol-permanence-ratio", type=float, default=EOL_PERMANENCE_RATIO,
                        help="Fracao da capacidade inicial acima da qual o cruzamento e anulado (modo absolute).")
    parser.add_argument("--eol-permanence-margin", type=float, default=0.10,
                        help="Margem somada ao limiar no modo margin.")
    parser.add_argument("--eol-permanence-on", choices=["raw", "smooth"], default="raw",
                        help="raw: checa a permanencia nos valores brutos (v2). smooth: na mediana movel.")
''', "argparse: flags v3, flag morta removida")

troca('''        eol_min_position=args.eol_min_position,
        drop_post_eol=not args.keep_post_eol,
    )''', '''        eol_min_position=args.eol_min_position,
        drop_post_eol=not args.keep_post_eol,
        eol_threshold=args.eol_threshold,
        eol_secondary_threshold=args.eol_secondary_threshold,
        permanence_mode=args.eol_permanence_mode,
        permanence_ratio=args.eol_permanence_ratio,
        permanence_margin=args.eol_permanence_margin,
        permanence_on_smooth=(args.eol_permanence_on == "smooth"),
    )''', "chamada de build_targets em main")

# ---------------------------------------------------------------- consumidores com nomes fixos
troca('''    battery_eol = (
        targets.groupby(["dataset", "phase_folder", "battery_id"], dropna=False)
        .agg(
            eol80_found=("target_eol80_found", "max"),
            eol70_found=("target_eol70_found", "max"),
        )
        .reset_index()
    )

    eol_counts = (
        battery_eol.groupby(["dataset", "phase_folder"], dropna=False)
        .agg(
            eol80_batteries=("eol80_found", "sum"),
            eol70_batteries=("eol70_found", "sum"),
        )
        .reset_index()
    )''', '''    # v3: os limiares sao descobertos pelas colunas presentes (eol80, eol85, ...)
    found_cols = sorted(c for c in targets.columns if re.fullmatch(r"target_eol\\d+_found", c))
    if not found_cols:
        raise ValueError("nenhuma coluna target_eolNN_found nos alvos")
    battery_eol = (
        targets.groupby(["dataset", "phase_folder", "battery_id"], dropna=False)
        .agg(**{c.replace("target_", ""): (c, "max") for c in found_cols})
        .reset_index()
    )

    eol_counts = (
        battery_eol.groupby(["dataset", "phase_folder"], dropna=False)
        .agg(**{c.replace("target_", "").replace("_found", "_batteries"): (c.replace("target_", ""), "sum")
                for c in found_cols})
        .reset_index()
    )''', "create_dataset_summary agnostico ao limiar")

troca('''        "capacity_end_to_initial_ratio",
        "eol80_found",
        "eol80_cycle_index",
        "eol70_found",
        "eol70_cycle_index",
        "trend_spearman",''', '''        "capacity_end_to_initial_ratio",
        *sorted(c for c in capacity_summary.columns if re.fullmatch(r"eol\\d+_(found|cycle_index|reject_reason)", c)),
        "trend_spearman",''', "preview do relatorio agnostico ao limiar")

# ---------------------------------------------------------------- sanidade
assert "allow_single_point_fallback" not in txt, "flag morta ainda presente"
DST.write_text(txt, encoding="utf-8")
py_compile.compile(str(DST), doraise=True)
print("\n".join(log)); print(f"\ncompilado: {DST.name}")
