# NASA PCoE Adapter v1.2

This package contains the reconstructed NASA PCoE adapter for the lithium-ion battery RUL project.

The workflow follows the project standard schema:

```text
src/adapters/nasa_pcoe/extract_pcoe_cycles.py
src/adapters/nasa_pcoe/clean_pcoe_cycles.py
src/adapters/nasa_pcoe/audit_pcoe_cycles.py
```

## Main changes in v1.2

Compared with v1.1, this version adds final model-usability guards for NASA PCoE:

1. Charge and discharge remain separated and use the same official clean-cycle schema.
2. Voltage hard bounds and terminal edge trimming from v1.1 are retained.
3. Cycles with very low phase capacity are kept in the clean files for traceability but are marked as not model-usable.
4. Cycles with phase_capacity_ah / initial_capacity_ah below the configured minimum are marked as not model-usable.
5. Discharge cycles below the EOL capacity before the assigned cycle_life are marked as not model-usable.
6. The audit now reports low-capacity and inconsistent pre-EOL capacity anomalies.
7. Additional model-usability thresholds are written to `cleaning_report.json` for traceability.
8. `tqdm` is used in extraction, cleaning, and auditing.

## Recommended execution

```powershell
python src/adapters/nasa_pcoe/extract_pcoe_cycles.py `
  --base-dir "D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries"

python src/adapters/nasa_pcoe/clean_pcoe_cycles.py `
  --base-dir "D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries" `
  --eol-ratio 0.80 `
  --eol-min-consecutive 2 `
  --eol-tail-fraction 0.80 `
  --min-eol-cycle 10 `
  --min-model-capacity-ah 0.20 `
  --min-model-capacity-ratio 0.50

python src/adapters/nasa_pcoe/audit_pcoe_cycles.py `
  --base-dir "D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries"
```

## Optional exploratory run

Use this only if you intentionally want to derive EOL from charge or another phase when discharge-derived EOL is unavailable:

```powershell
python src/adapters/nasa_pcoe/clean_pcoe_cycles.py `
  --base-dir "D:\USUÁRIOS\Marcellus\OneDrive\Doutorado\BATERIAS - Ressubmissão\Batteries" `
  --allow-phase-eol-fallback
```

For the article pipeline, the conservative default is preferred.
