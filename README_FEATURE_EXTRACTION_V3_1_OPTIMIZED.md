# Extração hierárquica v3.1 otimizada

Este pacote substitui apenas o script:

```text
src/feature_extraction/build_hierarchical_features_v3.py
```

## Alterações principais

1. `parse_vector` foi otimizado para evitar `pd.Series` em cada vetor. Esse era um gargalo forte em bases grandes como MATR.
2. A inclinação em relação ao tempo passou a usar fórmula fechada de regressão linear, em vez de `np.polyfit` para cada ciclo e variável.
3. A busca de targets por linha foi trocada por dicionário indexado por `source_row_id`, reduzindo overhead de `.loc` em loop.
4. Foi adicionada a flag `--fast-mode`.

## O que `--fast-mode` muda

Com `--fast-mode`, a coluna:

```text
<variavel>_corr_vs_time
```

passa a receber correlação de Pearson em vez de Spearman, para reduzir custo computacional. As colunas permanecem com o mesmo nome para não quebrar o pipeline downstream, mas o relatório/metodologia deve registrar essa diferença se esse modo for usado nos resultados finais.

## Comando recomendado para MATR

```powershell
cd "C:\Artigo Baterias de Litio\Batteries"

& C:\Users\Giovanny\anaconda3\envs\batteries\python.exe -u src\feature_extraction\build_hierarchical_features_v3.py `
  --input-root "C:\Artigo Baterias de Litio\Batteries\data\interim" `
  --output-root "C:\Artigo Baterias de Litio\Batteries\data\processed\features_hierarchical_v3" `
  --reports-root "C:\Artigo Baterias de Litio\Batteries\reports\feature_extraction_v3" `
  --dataset "MATR" `
  --phase-dirs charge `
  --workers 1 `
  --write-batch-size 5000 `
  --force `
  --no-stages `
  --fast-mode
```

Depois rode a descarga trocando `charge` por `discharge`.

## Observação

Para máxima estabilidade em 16 GB de RAM, rode uma fase por vez. Use `--workers 1`. Só teste `--workers 2` se a RAM ficar abaixo de 85% durante a execução.
