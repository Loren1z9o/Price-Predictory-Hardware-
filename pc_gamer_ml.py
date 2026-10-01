"""
PC Gamer Price Predictor — v7.0
Regressão supervisionada do preço (R$) de um PC a partir dos componentes.
Modelos: Regressão Linear Múltipla (A) e KNN Regressor (B).

Ordem do pipeline (Guia Aula 3):
  1. gerar_dataset()   dados brutos, nenhuma estatística calculada
  2. dividir()         split 70/15/15 estratificado ANTES de qualquer EDA
  3. resumo_eda()      EDA só no treino
  4. treinar()         encoders dentro de Pipelines (fit só no treino);
                       K do KNN escolhido por GridSearchCV (CV=5 no treino)
  5. seleção           menor MAE na VALIDAÇÃO
  6. teste             usado uma única vez, depois da seleção
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
TIERS = ["entry", "budget", "mid", "high", "enthusiast"]

# ── Catálogo (preços em R$, hardwarebarato.com / KaBuM / Pichau, ago/2026) ──
# Formato: nome: (preço, tier, [socket], [ddr])
CATALOGO = {
    "cpu": {
        "AMD Ryzen 5 5500":    (528,  "entry",      "AM4"),
        "AMD Ryzen 5 5600":    (600,  "entry",      "AM4"),
        "AMD Ryzen 7 5700X":   (1198, "budget",     "AM4"),
        "AMD Ryzen 7 5800X3D": (2150, "high",       "AM4"),
        "AMD Ryzen 5 8400F":   (645,  "entry",      "AM5"),
        "AMD Ryzen 5 7600":    (980,  "budget",     "AM5"),
        "AMD Ryzen 7 7700X":   (1500, "mid",        "AM5"),
        "AMD Ryzen 7 7800X3D": (1895, "high",       "AM5"),
        "AMD Ryzen 7 9800X3D": (2600, "enthusiast", "AM5"),
        "AMD Ryzen 9 9950X":   (3800, "enthusiast", "AM5"),
        "Intel i3-12100F":     (620,  "entry",      "LGA1700"),
        "Intel i5-12400F":     (790,  "entry",      "LGA1700"),
        "Intel i5-14400F":     (962,  "budget",     "LGA1700"),
        "Intel i5-14600KF":    (1400, "mid",        "LGA1700"),
        "Intel i7-14700KF":    (2100, "high",       "LGA1700"),
        "Intel i9-14900K":     (2898, "enthusiast", "LGA1700"),
    },
    "mobo": {
        "Gigabyte A520M K V2":       (355,  "entry",  "AM4",     "DDR4"),
        "MSI B550M-A Pro":           (481,  "budget", "AM4",     "DDR4"),
        "ASUS TUF B550M-Plus":       (650,  "mid",    "AM4",     "DDR4"),
        "ASUS Prime A620M-K":        (450,  "entry",  "AM5",     "DDR5"),
        "MSI PRO B850M-B":           (590,  "budget", "AM5",     "DDR5"),
        "MSI MAG B650 Tomahawk":     (1000, "mid",    "AM5",     "DDR5"),
        "MSI MAG X870E Tomahawk":    (2120, "high",   "AM5",     "DDR5"),
        "MSI PRO H610M-G DDR4":      (420,  "entry",  "LGA1700", "DDR4"),
        "MSI PRO B760M-A DDR4":      (600,  "budget", "LGA1700", "DDR4"),
        "ASUS TUF B760M-Plus DDR5":  (850,  "mid",    "LGA1700", "DDR5"),
        "MSI MAG Z790 Tomahawk DDR5": (1410, "high",  "LGA1700", "DDR5"),
    },
    "ram": {
        "DDR4 8GB 3200MHz (1x8)":    (220,  "entry",      None, "DDR4"),
        "DDR4 16GB 3200MHz (2x8)":   (390,  "budget",     None, "DDR4"),
        "DDR4 32GB 3200MHz (2x16)":  (700,  "mid",        None, "DDR4"),
        "DDR4 32GB 3600MHz RGB":     (1000, "high",       None, "DDR4"),
        "DDR5 16GB 5600MHz (2x8)":   (480,  "budget",     None, "DDR5"),
        "DDR5 32GB 5600MHz (2x16)":  (750,  "mid",        None, "DDR5"),
        "DDR5 32GB 6000MHz RGB":     (1000, "high",       None, "DDR5"),
        "DDR5 64GB 5600MHz (2x32)":  (2000, "enthusiast", None, "DDR5"),
    },
    "gpu": {
        "GTX 1650":         (850,   "entry"),
        "RX 6600":          (1200,  "entry"),
        "RTX 3060 12GB":    (1600,  "budget"),
        "RX 7600":          (1820,  "budget"),
        "Arc B580":         (2200,  "budget"),
        "RTX 4060":         (2300,  "budget"),
        "RX 7800 XT":       (3400,  "mid"),
        "RTX 5060 Ti 16GB": (3783,  "mid"),
        "RTX 4070":         (3800,  "mid"),
        "RTX 5070":         (4263,  "high"),
        "RX 9070 XT":       (4900,  "high"),
        "RTX 5070 Ti":      (6999,  "high"),
        "RTX 5080":         (9066,  "enthusiast"),
        "RTX 5090":         (18000, "enthusiast"),
    },
    "ssd": {
        "Kingston NV2 500GB":  (200,  "entry"),
        "Crucial P3 500GB":    (220,  "budget"),
        "Kingston NV2 1TB":    (350,  "budget"),
        "WD Blue SN580 1TB":   (380,  "mid"),
        "Samsung 990 Pro 1TB": (620,  "high"),
        "WD Black SN850X 2TB": (950,  "high"),
        "Samsung 990 Pro 4TB": (1900, "enthusiast"),
    },
    "fonte": {
        "Redragon RGPS 500W Bronze": (150,  "entry"),
        "Corsair CX650 Bronze":      (350,  "budget"),
        "Cooler Master MWE 650 Gold": (400, "budget"),
        "Corsair RM750x Gold":       (560,  "mid"),
        "Corsair RM850x Gold":       (900,  "high"),
        "Corsair RM1000x Gold":      (1100, "enthusiast"),
    },
    "cooler": {
        "Cooler box (stock)":        (0,    "entry"),
        "DeepCool AK400":            (180,  "budget"),
        "Thermalright Peerless Assassin": (280, "mid"),
        "DeepCool LE720 360mm":      (560,  "high"),
        "NZXT Kraken 360":           (1000, "enthusiast"),
    },
    "gabinete": {
        "Gamemax Infinit M909":    (170,  "entry"),
        "Corsair 3000D Airflow":   (370,  "budget"),
        "NZXT H5 Flow":            (480,  "mid"),
        "Lian Li O11 Dynamic EVO": (900,  "high"),
        "Corsair 7000D Airflow":   (1250, "enthusiast"),
    },
}
COLS = list(CATALOGO)                      # 8 features categóricas
NOMES_COLS = {"cpu": "Processador", "mobo": "Placa-mãe", "ram": "Memória RAM",
              "gpu": "Placa de vídeo", "ssd": "SSD", "fonte": "Fonte",
              "cooler": "Cooler", "gabinete": "Gabinete"}


def preco(col, item):
    return CATALOGO[col][item][0]


def itens_por_preco(col):
    """Itens da categoria do mais barato ao mais caro."""
    return sorted(CATALOGO[col], key=lambda k: CATALOGO[col][k][0])


def compativeis(col, socket=None, ddr=None):
    """Filtra placa-mãe por socket e RAM por DDR."""
    itens = CATALOGO[col]
    if col == "mobo":
        return [k for k, v in itens.items() if v[2] == socket]
    if col == "ram":
        return [k for k, v in itens.items() if v[3] == ddr]
    return list(itens)


# ── 1. Geração do dataset ──────────────────────────────────────────────
def _sorteia(rng, col, tier, opcoes):
    """Sorteia um item do tier da build ou de um tier vizinho."""
    i = TIERS.index(tier)
    ok = set(TIERS[max(0, i - 1): i + 2])
    cand = [k for k in opcoes if CATALOGO[col][k][1] in ok] or opcoes
    return cand[rng.integers(len(cand))]


def gerar_dataset(n=3000, seed=SEED):
    """Preço = soma dos componentes × variação de mercado de ±8%."""
    rng = np.random.default_rng(seed)
    linhas = []
    for _ in range(n):
        tier = TIERS[rng.choice(5, p=[0.12, 0.23, 0.30, 0.22, 0.13])]
        b = {"cpu": _sorteia(rng, "cpu", tier, list(CATALOGO["cpu"]))}
        b["mobo"] = _sorteia(rng, "mobo", tier,
                             compativeis("mobo", socket=CATALOGO["cpu"][b["cpu"]][2]))
        b["ram"] = _sorteia(rng, "ram", tier,
                            compativeis("ram", ddr=CATALOGO["mobo"][b["mobo"]][3]))
        for col in ["gpu", "ssd", "fonte", "cooler", "gabinete"]:
            b[col] = _sorteia(rng, col, tier, list(CATALOGO[col]))
        soma = sum(preco(c, b[c]) for c in COLS)
        linhas.append({**b, "tier": tier,
                       "preco": round(soma * rng.uniform(0.92, 1.08), 2)})
    return pd.DataFrame(linhas)


# ── 2. Split 70/15/15 (antes de qualquer estatística) ──────────────────
def dividir(df):
    X, y, t = df[COLS], df["preco"], df["tier"]
    X_tr, X_tmp, y_tr, y_tmp, t_tr, t_tmp = train_test_split(
        X, y, t, test_size=0.30, stratify=t, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=t_tmp, random_state=SEED)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ── 3. EDA (somente treino) ────────────────────────────────────────────
def resumo_eda(X_tr, y_tr, tier_tr):
    q1, q3 = y_tr.quantile([0.25, 0.75])
    iqr = q3 - q1
    return {
        "linhas_treino": len(X_tr), "colunas": len(COLS),
        "nulos": int(X_tr.isna().sum().sum() + y_tr.isna().sum()),
        "preco": {k: round(float(v), 2) for k, v in y_tr.describe().items()},
        "outliers_iqr": int(((y_tr < q1 - 1.5 * iqr) | (y_tr > q3 + 1.5 * iqr)).sum()),
        "limite_superior_iqr": round(float(q3 + 1.5 * iqr), 2),
        "builds_por_tier": tier_tr.value_counts().reindex(TIERS).to_dict(),
    }


# ── 4. Modelos ─────────────────────────────────────────────────────────
def pipe_linear():
    """One-Hot com drop='first' (evita a armadilha da variável dummy).
    As categorias vão em ordem de preço, então a coluna descartada é
    sempre o item MAIS BARATO: cada coeficiente vira o 'custo extra'
    de trocar o item mais barato por aquele item."""
    enc = OneHotEncoder(categories=[itens_por_preco(c) for c in COLS],
                        drop="first", sparse_output=False)
    return Pipeline([("prep", ColumnTransformer([("cat", enc, COLS)])),
                     ("modelo", LinearRegression())])


def pipe_knn():
    """One-Hot completo (sem drop): cada troca de componente pesa igual
    na distância. Duas builds que diferem em m componentes ficam a
    distância euclidiana √(2m)."""
    enc = OneHotEncoder(categories=[itens_por_preco(c) for c in COLS],
                        sparse_output=False)
    return Pipeline([("prep", ColumnTransformer([("cat", enc, COLS)])),
                     ("modelo", KNeighborsRegressor())])


def metricas(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p)),
            "MAPE": float(np.mean(np.abs((y - p) / y)) * 100)}


def treinar(n=3000):
    df = gerar_dataset(n)
    X_tr, X_va, X_te, y_tr, y_va, y_te = dividir(df)
    cv = KFold(5, shuffle=True, random_state=SEED)

    # Modelo A — Linear (sem hiperparâmetros a ajustar)
    lin = pipe_linear().fit(X_tr, y_tr)
    cv_lin = -cross_val_score(pipe_linear(), X_tr, y_tr, cv=cv,
                              scoring="neg_mean_absolute_error").mean()

    # Modelo B — KNN: GridSearchCV em K e no tipo de peso (só treino)
    grid = GridSearchCV(pipe_knn(),
                        {"modelo__n_neighbors": list(range(1, 31)),
                         "modelo__weights": ["uniform", "distance"]},
                        cv=cv, scoring="neg_mean_absolute_error").fit(X_tr, y_tr)
    knn = grid.best_estimator_
    r = grid.cv_results_
    curva_k = pd.DataFrame({"K": r["param_modelo__n_neighbors"].astype(int),
                            "pesos": r["param_modelo__weights"].astype(str),
                            "MAE_cv": -r["mean_test_score"]})

    # Baseline — DummyRegressor: prevê sempre a média do treino. Serve de piso:
    # um modelo só é útil se tiver MAE bem menor que este (não entra na seleção).
    dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    cv_dummy = -cross_val_score(DummyRegressor(strategy="mean"), X_tr, y_tr, cv=cv,
                                scoring="neg_mean_absolute_error").mean()

    modelos = {"Linear": lin, "KNN": knn}
    val = {m: metricas(y_va, p.predict(X_va)) for m, p in modelos.items()}
    melhor = min(val, key=lambda m: val[m]["MAE"])       # decisão pela validação
    teste = {m: metricas(y_te, p.predict(X_te)) for m, p in modelos.items()}  # uso único

    return {
        "df": df, "X_tr": X_tr, "y_tr": y_tr, "X_te": X_te, "y_te": y_te,
        "modelos": modelos, "melhor": melhor, "curva_k": curva_k,
        "knn_params": {"K": grid.best_params_["modelo__n_neighbors"],
                       "pesos": grid.best_params_["modelo__weights"]},
        "cv_mae": {"Linear": float(cv_lin), "KNN": float(-grid.best_score_)},
        "validacao": val, "teste": teste,
        "baseline": {"cv_mae": float(cv_dummy),
                     "validacao": metricas(y_va, dummy.predict(X_va)),
                     "teste": metricas(y_te, dummy.predict(X_te))},
        "pred_teste": {m: p.predict(X_te) for m, p in modelos.items()},
        "eda": resumo_eda(X_tr, y_tr, df.loc[X_tr.index, "tier"]),
        "split": {"treino": len(X_tr), "validacao": len(X_va), "teste": len(X_te)},
    }


# ── Explicabilidade (usada pela aba Teoria) ────────────────────────────
def coeficientes_linear(lin):
    """Tabela item → coeficiente aprendido × custo extra real."""
    nomes = lin[:-1].get_feature_names_out()
    coef = dict(zip(nomes, lin[-1].coef_))
    linhas = []
    for col in COLS:
        base = itens_por_preco(col)[0]
        for item in itens_por_preco(col):
            linhas.append({"categoria": col, "item": item,
                           "preco": preco(col, item),
                           "extra_real": preco(col, item) - preco(col, base),
                           "coef": float(coef.get(f"cat__{col}_{item}", 0.0)),
                           "referencia": item == base})
    return pd.DataFrame(linhas), float(lin[-1].intercept_)


def vizinhos_knn(knn, X_tr, y_tr, build):
    """Retorna os K vizinhos da build, distâncias e a predição manual."""
    q = pd.DataFrame([build])[COLS]
    dist, idx = knn[-1].kneighbors(knn[:-1].transform(q))
    dist, idx = dist[0], idx[0]
    viz = X_tr.iloc[idx].copy()
    viz["preco"] = y_tr.iloc[idx].values
    viz["distancia"] = dist
    viz["n_diferentes"] = np.rint(dist ** 2 / 2).astype(int)
    viz["difere_em"] = [", ".join(NOMES_COLS[c] for c in COLS if r[c] != build[c]) or "—"
                        for _, r in viz.iterrows()]
    if knn[-1].weights == "uniform" or (dist == 0).any():
        w = np.ones_like(dist) if not (dist == 0).any() else (dist == 0).astype(float)
    else:
        w = 1 / dist
    viz["peso"] = w / w.sum()
    manual = float((viz["peso"] * viz["preco"]).sum())
    return viz.reset_index(drop=True), manual


def salvar_json(res, caminho="dados.json"):
    out = {k: res[k] for k in ["melhor", "knn_params", "cv_mae", "validacao",
                               "teste", "baseline", "eda", "split"]}
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    res = treinar()
    print(f"Split: {res['split']} | KNN: {res['knn_params']}")
    for m in res["modelos"]:
        v, t = res["validacao"][m], res["teste"][m]
        print(f"{m:7s} CV MAE R${res['cv_mae'][m]:,.0f} | Val MAE R${v['MAE']:,.0f} "
              f"| Teste MAE R${t['MAE']:,.0f} R² {t['R2']:.3f} MAPE {t['MAPE']:.1f}%")
    b = res["baseline"]
    print(f"Baseline CV MAE R${b['cv_mae']:,.0f} | Val MAE R${b['validacao']['MAE']:,.0f} "
          f"| Teste MAE R${b['teste']['MAE']:,.0f} R² {b['teste']['R2']:.3f} (média do treino)")
    print("Melhor (validação):", res["melhor"])
    salvar_json(res)
