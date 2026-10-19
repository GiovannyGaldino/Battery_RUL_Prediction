#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATCH 02 - build_modeling_from_features_v3.py (matriz de modelagem)

Substitui a deny-list por nome (should_remove_column, default "mantem") por uma
LISTA DE INCLUSAO por familia de escopo intra-ciclo (default "rejeita").

  --column-policy allowlist        (default v3)  so entra o que esta declarado
  --column-policy legacy_denylist  reproduz v2 exatamente, para comparacao

Motivo: em v2 as cinco estatisticas de tendencia da trajetoria inteira
(capacity_trend_*, capacity_*_step_rate, capacity_median_abs_step_ah) passaram
porque o nome nao casava com nenhum padrao proibido. Uma lista de inclusao por
familia nao depende de antecipar nomes.

Tambem grava <branch>_<level>_column_policy.json com a politica e as listas.
"""
import sys, pathlib, py_compile
RAIZ = pathlib.Path(__file__).resolve().parent.parent
SRC = RAIZ / "pristine" / "build_modeling_from_features_v3.py"
DST = RAIZ / "patched" / "build_modeling_from_features_v3.py"
txt = SRC.read_text(encoding="utf-8"); log = []
def troca(velho, novo, nome):
    global txt
    n = txt.count(velho)
    if n != 1: sys.exit(f"ABORTADO [{nome}]: ancora ocorre {n}x")
    txt = txt.replace(velho, novo, 1); log.append(f"  ok  {nome}")

troca('''SUBSTRINGS_TO_REMOVE = (
    "rul",
    "remaining_cycles",
    "eol",
    "after_eol",
)
''', '''SUBSTRINGS_TO_REMOVE = (
    "rul",
    "remaining_cycles",
    "eol",
    "after_eol",
)

# ---------------------------------------------------------------------------
# v3: LISTA DE INCLUSAO por familia de escopo intra-ciclo.
#
# Uma coluna so vira preditor se (a) comecar com um prefixo permitido ou for um
# nome permitido, e (b) nao comecar com um prefixo proibido. O default e
# rejeitar. Colunas de escopo de celula (estatisticas da trajetoria inteira)
# chegam do re-join com prefixo cell_ e sao rejeitadas por construcao; as
# contrapartes causais em janela expansiva chegam com prefixo causal_.
# ---------------------------------------------------------------------------
ALLOWED_PREFIXES = (
    "voltage_", "current_", "temperature_", "power_", "energy_",
    "capacityh_", "time_", "integral_",
    "causal_",                     # tendencia em janela expansiva (so passado)
    "vq_", "qv_", "eq_",           # features de curva, quando presentes
)
ALLOWED_EXACT = {
    # capacity_* que sao intra-ciclo (regressao contra o tempo DENTRO do ciclo)
    "capacity_slope_vs_time",
    "capacity_corr_vs_time",
    "capacity_gap_jump_n",
    "capacity_abs_integral_minus_input_phase_capacity_ah",
}
FORBIDDEN_PREFIXES = (
    "cell_",       # escopo de celula: estatistica da trajetoria inteira
    "target_", "input_", "source_",
)


def should_remove_column_allowlist(col: str, target: str) -> tuple[bool, str]:
    """v3. Devolve (remove, motivo). Default: remove."""
    c = col.lower()
    if col == target:
        return False, "target_y"
    if col in IDENTIFIER_COLUMNS:
        return True, "identifier_or_group_column"
    if c.startswith(FORBIDDEN_PREFIXES):
        return True, "forbidden_prefix_" + c.split("_", 1)[0]
    if col in LEAKAGE_EXACT or any(s in c for s in SUBSTRINGS_TO_REMOVE):
        return True, "lifetime_label_or_substring"
    if col in ALLOWED_EXACT:
        return False, "allowlist_exact"
    if c.startswith(ALLOWED_PREFIXES):
        return False, "allowlist_prefix"
    return True, "not_in_allowlist"
''', "lista de inclusao")

troca('''    parser.add_argument(
        "--min-non-null-rate",
        type=float,
        default=0.20,
        help="Drop feature columns with less than this non-null rate.",
    )
    return parser.parse_args()''', '''    parser.add_argument(
        "--min-non-null-rate",
        type=float,
        default=0.20,
        help="Drop feature columns with less than this non-null rate.",
    )
    parser.add_argument(
        "--column-policy",
        choices=["allowlist", "legacy_denylist"],
        default="allowlist",
        help="allowlist (v3): so entram familias intra-ciclo declaradas. "
             "legacy_denylist: reproduz v2 (deny-list por nome).",
    )
    return parser.parse_args()''', "argparse --column-policy")

troca('''def should_remove_column(col: str, target: str) -> tuple[bool, str]:
    c = col.lower()
''', '''def should_remove_column(col: str, target: str, policy: str = "allowlist") -> tuple[bool, str]:
    if policy == "allowlist":
        return should_remove_column_allowlist(col, target)
    if policy != "legacy_denylist":
        raise ValueError(f"column policy invalida: {policy!r}")
    c = col.lower()
''', "despacho por politica")

troca('''    min_non_null_rate: float,
    output_root: Path,
) -> None:
    branch_files = [f for f in files if f.parts[-3] == branch]''', '''    min_non_null_rate: float,
    output_root: Path,
    column_policy: str = "allowlist",
) -> None:
    branch_files = [f for f in files if f.parts[-3] == branch]''', "assinatura build_branch")

troca('''        remove, reason = should_remove_column(col, target=target)
''', '''        remove, reason = should_remove_column(col, target=target, policy=column_policy)
''', "chamada com politica")

troca('''    pd.DataFrame(removed_rows).to_csv(out_removed, index=False)''', '''    pd.DataFrame(removed_rows).to_csv(out_removed, index=False)

    # v3: registra a politica e as listas usadas, para auditoria
    import json
    policy_path = output_root / f"{branch}_{feature_level}_column_policy.json"
    with open(policy_path, "w", encoding="utf-8") as fh:
        json.dump({
            "column_policy": column_policy,
            "allowed_prefixes": list(ALLOWED_PREFIXES),
            "allowed_exact": sorted(ALLOWED_EXACT),
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "n_features_kept": len(kept_features),
            "n_columns_removed": len(removed_rows),
        }, fh, indent=2, ensure_ascii=False)''', "column_policy.json")

troca('''            min_non_null_rate=args.min_non_null_rate,
            output_root=output_root,
        )''', '''            min_non_null_rate=args.min_non_null_rate,
            output_root=output_root,
            column_policy=args.column_policy,
        )''', "main passa a politica")

DST.write_text(txt, encoding="utf-8"); py_compile.compile(str(DST), doraise=True)
print("\n".join(log)); print(f"\ncompilado: {DST.name}")
