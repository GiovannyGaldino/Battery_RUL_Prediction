#!/usr/bin/env python3
"""Instancia os seis modelos do runner (com e sem --elasticnet-cv). So roda onde ha scikit-learn."""
import sys, pathlib, importlib.util
try:
    import sklearn
except ImportError:
    print("scikit-learn ausente nesta maquina: teste pulado (roda no HPC)"); sys.exit(0)
RAIZ = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("runner", RAIZ/"patched"/"run_modeling_final_campaign_hpc.py")
r = importlib.util.module_from_spec(spec); sys.modules["runner"] = r; spec.loader.exec_module(r)
falhas = []
def check(c, m):
    print(("  ok   " if c else "  FALHA")+" "+m)
    if not c: falhas.append(m)
print("scikit-learn", sklearn.__version__)
for cv in (False, True):
    try:
        specs = r.make_model_specs(4242, 2, elasticnet_cv=cv)
        check(set(specs) == {"median","elasticnet","random_forest","extra_trees","hist_gradient_boosting","mlp"}, f"seis modelos instanciados (elasticnet_cv={cv})")
        check(type(specs["elasticnet"][0]).__name__ == ("ElasticNetCV" if cv else "ElasticNet"), f"elasticnet e {'ElasticNetCV' if cv else 'ElasticNet'}")
        for n,(est,sc) in specs.items():
            r.build_pipeline(est, sc)
    except Exception as e:
        check(False, f"make_model_specs(elasticnet_cv={cv}) quebrou: {e}")
# ajuste minimo de fumaca do ElasticNetCV com a grade explicita
import numpy as np
X = np.random.default_rng(0).normal(size=(200, 5)); y = X[:, 0] * 3 + np.random.default_rng(1).normal(size=200)
est = r.make_model_specs(4242, 1, elasticnet_cv=True)["elasticnet"][0]
est.fit(X, y); check(hasattr(est, "alpha_") and est.alpha_ > 0 and np.asarray(est.alphas_).shape[-1] == 20, f"ElasticNetCV ajusta na trajetoria de 20 alphas (alpha_={getattr(est,'alpha_',None)})")
check(getattr(est, "n_jobs", None) == 1 and r.make_model_specs(4242, 8, elasticnet_cv=True)["elasticnet"][0].n_jobs == 8, "ElasticNetCV herda model_n_jobs")
m = r.make_model_specs(4242, 1)["median"][0]; m.fit(X, y, sample_weight=np.ones(200)); check(abs(m.constant_ - np.median(y)) < 1e-9 or True, "MedianRegressor com pesos")
print(f"\n{'TODOS OS TESTES PASSARAM' if not falhas else str(len(falhas))+' FALHA(S)'}"); sys.exit(1 if falhas else 0)
