import pandas as pd

# Converter para CSV
df_charge = pd.read_parquet(r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\processed\charge\charge_features_adaptive.parquet")
df_charge.to_csv("charge_features_adaptive.csv", index=False)

df_discharge = pd.read_parquet(r"D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries\data\processed\discharge\discharge_features_adaptive.parquet")
df_discharge.to_csv("discharge_features_adaptive.csv", index=False)