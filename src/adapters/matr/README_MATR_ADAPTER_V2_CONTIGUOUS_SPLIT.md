# MATR Adapter v2 — segmentação contínua por limiar de corrente

## Objetivo

Esta versão corrige a lógica de extração da base MATR/Severson para evitar a criação de sinais ruidosos causados por seleção não contínua de pontos.

A versão anterior separava as fases por máscara:

```text
charge    = todos os pontos com I > epsilon
discharge = todos os pontos com I < -epsilon
```

Isso podia remover trechos de baixa corrente/repouso e também juntar amostras separadas no tempo. A versão v2 acessa novamente os arquivos `.mat` brutos e mantém a estrutura temporal do ciclo completo.

## Nova regra de segmentação

Para cada ciclo completo da estrutura `.mat`:

```text
ponto_inicial_descarga = primeiro índice em que current_a < -0.5 A
```

A partir disso:

```text
charge    = do início do ciclo até antes de ponto_inicial_descarga
discharge = de ponto_inicial_descarga até o fim do ciclo
```

O limiar é parametrizável por:

```text
--discharge-current-threshold-a 0.5
```

Por padrão, a descarga começa no primeiro ponto com:

```text
I < -0.5 A
```

## Arquivos incluídos

```text
src/adapters/matr/extract_matr_cycles.py
src/adapters/matr/clean_matr_cycles.py
src/adapters/matr/audit_matr_cycles.py
scripts/pipelines/run_matr_v2_pipeline.py
```

## Principais mudanças

### `extract_matr_cycles.py`

- Lê novamente os `.mat` brutos com `h5py`.
- Acessa `batch -> cycles -> V, I, T, t, Qc, Qd`.
- Detecta o início da descarga pelo primeiro ponto com corrente menor que `-0.5 A`.
- Mantém carga e descarga como segmentos contíguos.
- Preserva os pontos de baixa corrente dentro da parte inicial de carga.
- Preserva todos os pontos da descarga a partir do limiar.
- Salva relatórios com o índice de início da descarga.

### `clean_matr_cycles.py`

- Compatível com a nova extração.
- Por padrão, não aplica janela de tensão, para preservar o ciclo dividido completo.
- Para reproduzir o comportamento antigo de corte por tensão, usar:

```text
--enable-voltage-window
```

## Como executar pelo terminal do VS Code

A partir da raiz do projeto:

```powershell
PS C:\Artigo Baterias de Litio\Batteries> & C:\Users\Giovanny\anaconda3\envs\batteries\python.exe scripts\pipelines\run_matr_v2_pipeline.py --base-dir "C:\Artigo Baterias de Litio\Batteries"
```

Se o projeto/código estiver no OneDrive, mas os dados estiverem no disco C, use o mesmo comando a partir da pasta do projeto. O `--base-dir` indica a raiz dos dados/relatórios no disco C; o runner localiza automaticamente os scripts a partir da pasta onde ele está salvo.

```powershell
PS D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries> & C:\Users\Giovanny\anaconda3\envs\batteries\python.exe scripts\pipelines\run_matr_v2_pipeline.py --base-dir "C:\Artigo Baterias de Litio\Batteries"
```

Se precisar informar explicitamente a pasta do código, use também `--project-dir`:

```powershell
& C:\Users\Giovanny\anaconda3\envs\batteries\python.exe scripts\pipelines\run_matr_v2_pipeline.py --project-dir "D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries" --base-dir "C:\Artigo Baterias de Litio\Batteries"
```

## Execução manual por etapas

### 1. Extração

```powershell
& C:\Users\Giovanny\anaconda3\envs\batteries\python.exe src\adapters\matr\extract_matr_cycles.py --base-dir "C:\Artigo Baterias de Litio\Batteries" --discharge-current-threshold-a 0.5
```

Saídas:

```text
C:\Artigo Baterias de Litio\Batteries\data\interim\matr\extracted\matr_charge_extracted_cycles.parquet
C:\Artigo Baterias de Litio\Batteries\data\interim\matr\extracted\matr_discharge_extracted_cycles.parquet
C:\Artigo Baterias de Litio\Batteries\data\interim\matr\extracted\matr_extraction_report.csv
C:\Artigo Baterias de Litio\Batteries\data\interim\matr\extracted\matr_extraction_report.json
```

### 2. Limpeza

```powershell
& C:\Users\Giovanny\anaconda3\envs\batteries\python.exe src\adapters\matr\clean_matr_cycles.py --base-dir "C:\Artigo Baterias de Litio\Batteries"
```

Saídas:

```text
C:\Artigo Baterias de Litio\Batteries\data\interim\matr\clean\matr_charge_clean_cycles.parquet
C:\Artigo Baterias de Litio\Batteries\data\interim\matr\clean\matr_discharge_clean_cycles.parquet
```

### 3. Auditoria

```powershell
& C:\Users\Giovanny\anaconda3\envs\batteries\python.exe src\adapters\matr\audit_matr_cycles.py --base-dir "C:\Artigo Baterias de Litio\Batteries"
```

## Comandos úteis

Exigir três pontos consecutivos abaixo de `-0.5 A` para evitar disparo por ruído pontual:

```powershell
& C:\Users\Giovanny\anaconda3\envs\batteries\python.exe src\adapters\matr\extract_matr_cycles.py --base-dir "C:\Artigo Baterias de Litio\Batteries" --confirm-discharge-points 3
```

Ativar corte por janela de tensão na limpeza:

```powershell
& C:\Users\Giovanny\anaconda3\envs\batteries\python.exe src\adapters\matr\clean_matr_cycles.py --base-dir "C:\Artigo Baterias de Litio\Batteries" --enable-voltage-window
```

## Observação metodológica

A nova extração não tenta identificar múltiplos estágios internos. Ela aplica apenas a regra definida para o MATR neste momento: dividir o ciclo completo em carga e descarga a partir do primeiro ponto em que a corrente fica menor que `-0.5 A`. Estágios, repousos e saltos temporais preservados no vetor devem ser tratados posteriormente pela etapa de extração hierárquica de características.

## Atualização v2.1 — escrita Parquet com menor uso de memória

A limpeza MATR pode gerar arquivos `*_clean_cycles.parquet` muito grandes, pois cada linha contém vetores (`time_s`, `voltage_v`, `current_a`, `temperature_c`, `capacity_ah`, `energy_wh`, `power_w`).

Na versão v2.1, `clean_matr_cycles.py` foi alterado para gravar o Parquet com `fastparquet` em pequenos row groups, controlados por:

```powershell
--parquet-row-group-size 500
```

Se ainda ocorrer `MemoryError`, usar:

```powershell
--parquet-row-group-size 100
```

ou, em computadores com pouca memória disponível:

```powershell
--parquet-row-group-size 50
```

A versão v2.1 também deixa de manter o dataframe limpo de carga inteiro armazenado em `results` enquanto processa a descarga, reduzindo o pico de memória entre fases.
