#!/usr/bin/env bash
# Caminhos da campanha v3 no HPC. Fonte unica: todos os passos leem daqui.
# Nada de v2 e sobrescrito: v3 escreve so em pastas novas.
export V3="${V3:-$HOME/campanha_v3}"                       # este pacote, extraido
export REPO="${REPO:-$HOME/projects/Batteries}"            # codigo v2 (pristine)
export INTERIM="${INTERIM:-$HOME/data/interim}"
export FEAT_V2="${FEAT_V2:-$HOME/data/processed/features_hierarchical_v3_fix}"
# Curvas V(Q)/Q(V): a pasta `_fixed` e a que a campanha v2 usou e a unica com curvas
# validas em todos os dominios. `curve_features_v3` (sem _fixed) so tem MATR e foi a
# causa do descarte da primeira campanha v3 em 2026-09-03. O portao 04 confere.
export CURVES="${CURVES:-$HOME/data/batteries/curve_features_v3_fixed}"
export DATA_V3="${DATA_V3:-$HOME/data/processed/v3}"
export OUT_V3="${OUT_V3:-$HOME/outputs/campanha_v3}"
export DATASETS="CALCE,HUST,MATR,NRW,ORA,PCOE,XJTU"            # nomes das pastas do interim (extracao)
export DATASETS_X="CALCE,HUST,MATR,NRW,OXFORD,PCOE,XJTU"       # nomes na coluna dataset da matriz
export SEED_SPLIT=42
export SEED_MODEL=4242
# O conteiner mostra 128 nucleos (nproc) mas entrega 32 (cgroup cpu.max = 3200000/100000,
# medido em 2026-09-03). 14 shards x 2 threads + braco B = 30 threads, dentro da cota.
export MODEL_N_JOBS="${MODEL_N_JOBS:-2}"
# Limita BLAS (ElasticNet, MLP) e OpenMP (HistGradientBoosting) ao mesmo teto por
# processo. Sem isso cada processo abre uma thread por nucleo (128), as threads
# fazem espera ativa e gastam a cota sem produzir; a carga passou de 600.
export OMP_NUM_THREADS="$MODEL_N_JOBS" OPENBLAS_NUM_THREADS="$MODEL_N_JOBS" MKL_NUM_THREADS="$MODEL_N_JOBS"
PATCHED="$V3/patched"; NEW="$V3/new"
mkdir -p "$DATA_V3" "$OUT_V3"
