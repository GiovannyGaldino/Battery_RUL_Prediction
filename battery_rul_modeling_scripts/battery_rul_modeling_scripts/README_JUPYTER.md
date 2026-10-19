# Battery RUL Modeling v3 — scripts para JupyterLab/Linux

Esta pasta contém scripts para continuar a modelagem de RUL a partir dos quatro arquivos já preparados:

```text
~/outputs/batteries_modeling_v3/modeling_ready/charge_X.parquet
~/outputs/batteries_modeling_v3/modeling_ready/charge_y.parquet
~/outputs/batteries_modeling_v3/modeling_ready/discharge_X.parquet
~/outputs/batteries_modeling_v3/modeling_ready/discharge_y.parquet
```

Os scripts **não refazem a preparação dos dados** nem a extração de características. Eles começam diretamente dos parquet de modelagem.

---

## 1. Arquivos principais

```text
run_modeling_v3.py              # treino principal mixed-dataset com split por bateria
run_lodo_modeling_v3.py         # validação leave-one-dataset-out opcional
inspect_modeling_inputs.py      # inspeção dos parquet antes do treino
battery_rul_modeling/           # pacote com funções auxiliares
scripts/run_inspection.sh       # comando pronto para inspeção
scripts/run_modeling_quick.sh   # primeira execução rápida
scripts/run_modeling_ablation.sh# execução com ablações
scripts/run_lodo_quick.sh       # LODO rápido opcional
```

---

## 2. Como usar no JupyterLab

Envie a pasta compactada para o JupyterLab e descompacte em uma pasta de trabalho, por exemplo:

```bash
cd ~/projects/Batteries
unzip battery_rul_modeling_scripts.zip
cd battery_rul_modeling_scripts
```

Faça primeiro a inspeção dos arquivos:

```bash
bash scripts/run_inspection.sh
```

Depois rode uma versão rápida da modelagem:

```bash
bash scripts/run_modeling_quick.sh
```

A saída principal será criada em:

```text
~/outputs/batteries_modeling_v3/modeling_results
```

---

## 3. Pipeline implementado

O pipeline segue a modelagem definida no artigo:

1. Carrega `charge_X`, `charge_y`, `discharge_X` e `discharge_y`.
2. Preserva metadados para auditoria e split: `dataset`, `battery_id`, `cycle_number`, `cycle_index`, `phase`, etc.
3. Remove metadados, identificadores e colunas com risco de vazamento da matriz preditiva.
4. Divide os dados por `battery_id`, evitando que ciclos da mesma bateria apareçam simultaneamente em treino, validação e teste.
5. Mantém a separação formal entre carga e descarga.
6. Treina os mesmos modelos para `charge` e `discharge`.
7. Usa validação cruzada agrupada por bateria dentro do treino para busca de hiperparâmetros.
8. Avalia no teste isolado por bateria.
9. Gera métricas globais, por dataset, por bateria e por faixa de RUL.
10. Gera tabelas e gráficos para comparar carga versus descarga.

---

## 4. Modelos implementados

A execução padrão usa:

```text
median                  # baseline ingênuo
elasticnet              # baseline linear regularizado
random_forest           # ensemble por bagging
extra_trees             # ensemble extremamente aleatorizado
hist_gradient_boosting  # boosting eficiente para dados tabulares
mlp                     # rede neural densa compacta
```

Esses modelos correspondem ao conjunto inicial descrito no artigo: baseline, modelo linear regularizado, ensembles de árvores, boosting e MLP compacta.

---

## 5. Métricas geradas

Para cada fase, experimento e modelo, são calculadas:

```text
MAE
RMSE
R²
sMAPE
MAPE filtrado
erro absoluto mediano
erro médio assinado
nMAE
nRMSE
erro máximo absoluto
```

Também são geradas métricas estratificadas por:

```text
dataset
battery_id
faixa absoluta de RUL: 0–50, 50–100, 100–200, 200–500, >500
faixa normalizada de RUL: late, middle, early
```

---

## 6. Estrutura de saída

```text
modeling_results/
├── charge/
│   ├── models/
│   ├── predictions/
│   ├── metrics/
│   ├── plots/
│   └── tables/
├── discharge/
│   ├── models/
│   ├── predictions/
│   ├── metrics/
│   ├── plots/
│   └── tables/
├── comparison/
│   ├── global_metrics_all_phases.csv
│   ├── charge_vs_discharge_metrics.csv
│   ├── model_ranking.csv
│   ├── dataset_error_comparison.csv
│   ├── battery_error_comparison.csv
│   ├── rul_abs_bin_error_comparison.csv
│   └── rul_norm_bin_error_comparison.csv
└── logs/
```

---

## 7. Execução principal completa

Depois que a execução rápida funcionar, rode a versão principal:

```bash
python run_modeling_v3.py \
  --input-root ~/outputs/batteries_modeling_v3/modeling_ready \
  --output-root ~/outputs/batteries_modeling_v3/modeling_results \
  --target target_rul80_steps_clipped \
  --phases charge discharge \
  --models median elasticnet random_forest extra_trees hist_gradient_boosting mlp \
  --experiments all_features \
  --n-iter 12 \
  --cv-folds 3 \
  --n-jobs -1
```

---

## 8. Experimentos de ablação

Para rodar as ablações descritas no texto metodológico:

```bash
bash scripts/run_modeling_ablation.sh
```

Experimentos disponíveis:

```text
all_features
no_curves
global_only
capacity_only
curve_only
no_temperature
no_energy
no_stage_gap
resolution_p003
resolution_p009
resolution_p015
resolution_p030
```

---

## 9. Leave-one-dataset-out opcional

O script `run_lodo_modeling_v3.py` executa validação LODO. Em cada rodada, um dataset completo fica fora do treino e é usado como teste.

Execução rápida:

```bash
bash scripts/run_lodo_quick.sh
```

Para selecionar datasets específicos:

```bash
python run_lodo_modeling_v3.py \
  --input-root ~/outputs/batteries_modeling_v3/modeling_ready \
  --output-root ~/outputs/batteries_modeling_v3/modeling_results_lodo \
  --target target_rul80_steps_clipped \
  --heldout-datasets HUST MATR XJTU \
  --quick
```

---

## 10. Observações importantes

- `cycle_number`, `cycle_index`, `dataset`, `battery_id`, `phase`, `source_file`, `source_path` e outros identificadores são removidos por padrão da matriz de entrada.
- Colunas com `target_`, `rul`, `remaining_cycles`, `eol`, `cycle_life` e termos similares também são removidas para evitar vazamento.
- O script mantém esses campos nas tabelas de predição e nos relatórios de erro.
- Modelos sensíveis à escala, como ElasticNet e MLP, usam `StandardScaler` dentro do pipeline.
- A remoção de colunas com muitos NaN e constantes é ajustada apenas no treino, dentro do pipeline.
- O MAPE convencional é reportado apenas como `mape_filtered`, usando um limiar mínimo de RUL real. A métrica percentual principal é o sMAPE.

---

## 11. Dependências

As dependências principais são:

```text
numpy
pandas
scikit-learn
joblib
matplotlib
pyarrow
```

Se faltar alguma dependência no ambiente:

```bash
pip install numpy pandas scikit-learn joblib matplotlib pyarrow
```
