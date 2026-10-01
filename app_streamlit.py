"""PC Gamer Price Predictor v7.0 — interface Streamlit.
Executar:  python -m streamlit run app_streamlit.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pc_gamer_ml as ml
from tab_teoria import render_teoria

st.set_page_config(page_title="PC Gamer Price Predictor", page_icon="🖥️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@500&display=swap');
h1, h2, h3 {font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em;}
[data-testid="stMetricValue"] {font-family: 'JetBrains Mono', monospace;}
.sub {color: #8a8f98; margin-top: -0.8rem;}
</style>""", unsafe_allow_html=True)

COR = {"Linear": "#2563eb", "KNN": "#f59e0b"}
brl = lambda v: f"R$ {v:,.0f}".replace(",", ".")


@st.cache_resource(show_spinner="Treinando Linear e KNN (só na primeira execução)…")
def carregar():
    return ml.treinar()


res = carregar()
st.title("🖥️ PC Gamer Price Predictor")
st.markdown('<p class="sub">Regressão Linear × KNN estimando o preço de um PC em reais '
            '· Machine Learning Clássico · UniSATC 2026</p>', unsafe_allow_html=True)

aba_pred, aba_res, aba_teo = st.tabs(["🎯 Preditor", "📊 Resultados", "📘 Teoria"])

# ── Preditor ───────────────────────────────────────────────────────────
with aba_pred:
    padrao = {"cpu": "AMD Ryzen 5 7600", "gpu": "RTX 4060", "ssd": "Kingston NV2 1TB",
              "fonte": "Corsair CX650 Bronze", "cooler": "DeepCool AK400",
              "gabinete": "Corsair 3000D Airflow"}
    sel = lambda col, opcoes, key: st.selectbox(
        ml.NOMES_COLS[col], opcoes, key=key,
        index=opcoes.index(padrao[col]) if padrao.get(col) in opcoes else 0,
        )  # sem preço na lista: o preço é o que os modelos precisam estimar

    c1, c2 = st.columns(2)
    with c1:
        build = {"cpu": sel("cpu", ml.itens_por_preco("cpu"), "cpu")}
        socket = ml.CATALOGO["cpu"][build["cpu"]][2]
        mobos = [k for k in ml.itens_por_preco("mobo") if k in ml.compativeis("mobo", socket=socket)]
        build["mobo"] = sel("mobo", mobos, "mobo_" + socket)
        ddr = ml.CATALOGO["mobo"][build["mobo"]][3]
        rams = [k for k in ml.itens_por_preco("ram") if k in ml.compativeis("ram", ddr=ddr)]
        build["ram"] = sel("ram", rams, "ram_" + ddr)
        build["gpu"] = sel("gpu", ml.itens_por_preco("gpu"), "gpu")
    with c2:
        for col in ["ssd", "fonte", "cooler", "gabinete"]:
            build[col] = sel(col, ml.itens_por_preco(col), col)
        st.caption(f"Compatibilidade aplicada: socket **{socket}** · memória **{ddr}**")

    X = pd.DataFrame([build])[ml.COLS]
    tabela = sum(ml.preco(c, build[c]) for c in ml.COLS)
    st.divider()
    m1, m2, m3 = st.columns(3)
    for col, nome in [(m1, "Linear"), (m2, "KNN")]:
        p = float(res["modelos"][nome].predict(X)[0])
        titulo = f"{nome} {'🏆' if nome == res['melhor'] else ''}"
        col.metric(titulo, brl(p), f"{(p - tabela) / tabela:+.1%} vs tabela", delta_color="off")
    m3.metric("Soma de tabela (referência)", brl(tabela))
    st.caption("A soma de tabela é o preço sem a variação de mercado de ±8% que existe no "
               "dataset. Veja na aba **Teoria** como cada modelo chegou ao seu número.")

# ── Resultados ─────────────────────────────────────────────────────────
with aba_res:
    linhas = []
    for m in res["modelos"]:
        v, t = res["validacao"][m], res["teste"][m]
        linhas.append({"Modelo": m, "MAE CV (treino)": brl(res["cv_mae"][m]),
                       "MAE validação": brl(v["MAE"]), "MAE teste": brl(t["MAE"]),
                       "RMSE teste": brl(t["RMSE"]), "R² teste": f"{t['R2']:.3f}",
                       "MAPE teste": f"{t['MAPE']:.1f}%"})
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
    s = res["split"]
    st.caption(f"Split estratificado 70/15/15: {s['treino']} treino · {s['validacao']} validação · "
               f"{s['teste']} teste. O vencedor (**{res['melhor']}**) foi escolhido pelo MAE de "
               f"validação; o teste foi calculado uma única vez, depois dessa escolha.")

    g1, g2 = st.columns(2)
    y_te = res["y_te"].values
    fig = go.Figure()
    for m, p in res["pred_teste"].items():
        fig.add_scatter(x=y_te, y=p, mode="markers", name=m, opacity=0.6,
                        marker=dict(size=5, color=COR[m]))
    lim = [y_te.min(), y_te.max()]
    fig.add_scatter(x=lim, y=lim, mode="lines", name="Ideal (y = ŷ)",
                    line=dict(dash="dash", color="gray"))
    fig.update_layout(title="Real × previsto (teste)", xaxis_title="Preço real (R$)",
                      yaxis_title="Preço previsto (R$)", height=420)
    g1.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for m, p in res["pred_teste"].items():
        fig.add_histogram(x=y_te - p, name=m, opacity=0.65, marker_color=COR[m], nbinsx=40)
    fig.update_layout(title="Distribuição dos resíduos (real − previsto)", barmode="overlay",
                      xaxis_title="Resíduo (R$)", yaxis_title="Builds", height=420)
    g2.plotly_chart(fig, width="stretch")

    with st.expander("🔎 Análise exploratória (somente conjunto de treino)"):
        e = res["eda"]
        a, b, c, d = st.columns(4)
        a.metric("Linhas (treino)", e["linhas_treino"])
        b.metric("Features", e["colunas"])
        c.metric("Valores nulos", e["nulos"])
        d.metric("Outliers (IQR)", e["outliers_iqr"])
        tr = res["df"].loc[res["X_tr"].index]
        fig = go.Figure()
        for t in ml.TIERS:
            fig.add_histogram(x=tr.loc[tr.tier == t, "preco"], name=t, nbinsx=30)
        fig.update_layout(barmode="stack", height=360, xaxis_title="Preço (R$)",
                          title="Preço por faixa de build (treino)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Os {e['outliers_iqr']} outliers acima de {brl(e['limite_superior_iqr'])} são "
                   "builds enthusiast reais (ex.: RTX 5090), por isso não foram removidos.")

# ── Teoria ─────────────────────────────────────────────────────────────
with aba_teo:
    render_teoria(res, build)
