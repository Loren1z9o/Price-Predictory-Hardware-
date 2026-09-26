"""Gera as figuras (Matplotlib) e o report_data.json usados no relatório.
Uso:  cd relatorio  &&  python gerar_figuras.py  &&  node build.js
"""
import json, os, sys
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))   # importa pc_gamer_ml da raiz
os.chdir(AQUI)                              # figuras e JSON ficam em relatorio/
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pc_gamer_ml as ml

os.makedirs("fig", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
AZ, LA = "#2563eb", "#f59e0b"
res = ml.treinar()
ml.salvar_json(res)

# Fig 1 — distribuição do preço no treino por faixa
tr = res["df"].loc[res["X_tr"].index]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.hist([tr.loc[tr.tier == t, "preco"] for t in ml.TIERS], bins=40, stacked=True,
        label=ml.TIERS, color=["#86efac", "#93c5fd", "#fde047", "#fdba74", "#fca5a5"])
ax.axvline(res["eda"]["limite_superior_iqr"], ls="--", c="k", lw=1)
ax.text(res["eda"]["limite_superior_iqr"] * 1.02, ax.get_ylim()[1] * 0.85, "limite IQR", fontsize=8)
ax.set_xlabel("Preço da build (R$)"); ax.set_ylabel("Nº de builds"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("fig/f1_eda.png"); plt.close()

# Fig 2 — coeficientes da Linear × custo extra real
coefs, b0 = ml.coeficientes_linear(res["modelos"]["Linear"])
c = coefs[~coefs.referencia]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for cat in ml.COLS:
    d = c[c.categoria == cat]
    ax.scatter(d.extra_real, d.coef, s=22, label=ml.NOMES_COLS[cat])
m = c.extra_real.max(); ax.plot([0, m], [0, m], "--", c="gray", lw=1, label="β = custo real")
ax.set_xlabel("Custo extra real em relação à peça mais barata (R$)")
ax.set_ylabel("Coeficiente aprendido β (R$)"); ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig("fig/f2_coef.png"); plt.close()

# Fig 3 — curva de K (GridSearchCV)
cv = res["curva_k"]; K = res["knn_params"]["K"]
fig, ax = plt.subplots(figsize=(6.5, 3.0))
for p, cor in [("uniform", "#94a3b8"), ("distance", LA)]:
    d = cv[cv.pesos == p]; ax.plot(d.K, d.MAE_cv, "o-", ms=3, c=cor, label=f"pesos = {p}")
ax.axvline(K, ls="--", c="k", lw=1); ax.text(K + 0.4, ax.get_ylim()[1] * 0.97, f"K = {K}", fontsize=8)
ax.set_xlabel("K (nº de vizinhos)"); ax.set_ylabel("MAE médio na CV (R$)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("fig/f3_k.png"); plt.close()

# Fig 4 — real × previsto no teste / Fig 5 — resíduos
y = res["y_te"].values
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.1), sharey=True)
for ax, (mod, cor) in zip(axs, [("Linear", AZ), ("KNN", LA)]):
    ax.scatter(y, res["pred_teste"][mod], s=6, alpha=0.5, c=cor)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "--", c="gray", lw=1)
    ax.set_title(mod, fontsize=10); ax.set_xlabel("Preço real (R$)")
axs[0].set_ylabel("Preço previsto (R$)")
fig.tight_layout(); fig.savefig("fig/f4_pred.png"); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 2.9))
for mod, cor in [("KNN", LA), ("Linear", AZ)]:
    ax.hist(y - res["pred_teste"][mod], bins=50, alpha=0.65, color=cor, label=mod)
ax.axvline(0, c="k", lw=0.8); ax.set_xlabel("Resíduo: real − previsto (R$)")
ax.set_ylabel("Nº de builds"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("fig/f5_res.png"); plt.close()

# Exemplo resolvido (mesma build padrão do app)
build = {"cpu": "AMD Ryzen 5 7600", "mobo": "ASUS Prime A620M-K", "ram": "DDR5 16GB 5600MHz (2x8)",
         "gpu": "RTX 4060", "ssd": "Kingston NV2 1TB", "fonte": "Corsair CX650 Bronze",
         "cooler": "DeepCool AK400", "gabinete": "Corsair 3000D Airflow"}
X = pd.DataFrame([build])[ml.COLS]
ref = coefs.set_index(["categoria", "item"])
lin_rows = [[ml.NOMES_COLS[k], build[k], float(ref.loc[(k, build[k])].extra_real),
             float(ref.loc[(k, build[k])].coef)] for k in ml.COLS]
viz, manual = ml.vizinhos_knn(res["modelos"]["KNN"], res["X_tr"], res["y_tr"], build)

out = {k: res[k] for k in ["melhor", "knn_params", "cv_mae", "validacao", "teste", "eda", "split"]}
out.update({
    "n_total": len(res["df"]), "b0": b0,
    "b0_real": sum(ml.preco(k, ml.itens_por_preco(k)[0]) for k in ml.COLS),
    "coef_erro_medio": float((c.coef - c.extra_real).abs().mean()),
    "coef_corr": float(np.corrcoef(c.coef, c.extra_real)[0, 1]),
    "n_itens": {k: len(v) for k, v in ml.CATALOGO.items()}, "n_onehot": int(len(coefs)),
    "gpu_5090": [float(ref.loc[("gpu", "RTX 5090")].extra_real), float(ref.loc[("gpu", "RTX 5090")].coef)],
    "ex_build": build, "ex_tabela": sum(ml.preco(k, build[k]) for k in ml.COLS),
    "ex_lin_rows": lin_rows, "ex_lin_pred": float(res["modelos"]["Linear"].predict(X)[0]),
    "ex_knn_rows": viz[["distancia", "n_diferentes", "difere_em", "preco", "peso"]].values.tolist(),
    "ex_knn_manual": manual, "ex_knn_pred": float(res["modelos"]["KNN"].predict(X)[0]),
})
json.dump(out, open("report_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: out[k] for k in ["b0", "b0_real", "coef_erro_medio", "coef_corr", "gpu_5090",
                                      "ex_tabela", "ex_lin_pred", "ex_knn_pred", "n_onehot"]}, default=float))
print(out["eda"])
