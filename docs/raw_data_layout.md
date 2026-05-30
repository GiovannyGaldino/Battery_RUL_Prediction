# Raw data layout

The repository does not redistribute raw datasets. Download each dataset from its official provider and preserve the original file structure as much as possible. Then place the files under `data/raw` using the folder names below.

| Dataset key | Expected local folder or file | Expected raw pattern |
|---|---|---|
| `calce` | `data/raw/CALCE/` | Excel workbooks, often grouped by cell ID (`**/*.xlsx`) |
| `hust` | `data/raw/HUST/` | HUST pickle files (`**/*.pkl`, `**/*.pickle`, or `**/*.pk`) |
| `matr` | `data/raw/MATR/` | MATR/Severson MATLAB files (`*.mat`) |
| `nasa_pcoe` | `data/raw/PCOE/` | NASA PCoE MATLAB files (`*.mat`) |
| `nrw` | `data/raw/NRW/` | NASA Randomized Battery Usage files (`RW*.mat`) |
| `ora` | `data/raw/ORA/Oxford_Battery_Degradation_Dataset_1.mat` | Oxford/ORA MATLAB file |
| `rwth` | `data/raw/RWTH/` | RWTH ZIP archives (`*.zip`) |
| `xjtu` | `data/raw/XJTU/` | XJTU MATLAB files, usually under batch folders (`**/*.mat`) |

If a provider distributes compressed archives, extract them only when the corresponding adapter expects extracted files. The RWTH adapter expects ZIP files directly.

Use explicit command-line arguments such as `--raw-dir` or `--mat-file` whenever your local layout differs from the table.
