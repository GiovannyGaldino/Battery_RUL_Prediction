#!/usr/bin/env python3
"""Confere as dependencias antes de qualquer passo. Sai com 1 se faltar algo."""
import importlib.util, sys
faltam = [m for m in ["numpy", "pandas", "pyarrow", "fastparquet", "sklearn", "tqdm", "joblib", "scipy"]
          if importlib.util.find_spec(m) is None]
print("python", sys.version.split()[0], "| faltam:", faltam or "nenhum")
if faltam:
    print("instale com:  pip install " + " ".join("scikit-learn" if m == "sklearn" else m for m in faltam))
    sys.exit(1)
