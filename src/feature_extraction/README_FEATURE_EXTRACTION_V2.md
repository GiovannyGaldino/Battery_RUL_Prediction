# Hierarchical Feature Extraction v2

Arquivos incluídos:

- `src/feature_extraction/build_hierarchical_features.py`: script principal de extração hierárquica.
- `scripts/run_feature_extraction_v2.py`: executor auxiliar para rodar uma base, uma bateria ou todas.

## Comando recomendado: todas as bases

```bat
C:\Users\Giovanny\anaconda3\envs\batteries\python.exe src\feature_extraction\build_hierarchical_features.py ^
  --data-root "C:\Artigo Baterias de Litio\Batteries\data" ^
  --phase all ^
  --dataset-filter all ^
  --feature-level H6 ^
  --batch-size 50 ^
  --resume ^
  --merge ^
  --parquet-engine fastparquet
```

## Comando recomendado: uma base

```bat
C:\Users\Giovanny\anaconda3\envs\batteries\python.exe src\feature_extraction\build_hierarchical_features.py ^
  --data-root "C:\Artigo Baterias de Litio\Batteries\data" ^
  --phase all ^
  --dataset-filter calce ^
  --feature-level H6 ^
  --batch-size 50 ^
  --resume ^
  --merge ^
  --parquet-engine fastparquet
```

## Usando o runner

```bat
C:\Users\Giovanny\anaconda3\envs\batteries\python.exe scripts\run_feature_extraction_v2.py ^
  --dataset all ^
  --phase all ^
  --resume ^
  --merge
```

## Saída

Os arquivos são salvos em:

```text
C:\Artigo Baterias de Litio\Batteries\data\processed\feature_extraction
C:\Artigo Baterias de Litio\Batteries\data\reports\feature_extraction
```

A saída usa blocos `part_*.parquet` por nível, fase, dataset e bateria. O modo `--resume` pula partes já concluídas.
