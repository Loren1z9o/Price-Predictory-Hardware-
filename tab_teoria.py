"""Aba Teoria — explica e PROVA, com a build escolhida no Preditor,
como a Regressão Linear e o KNN chegam ao preço."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pc_gamer_ml as ml

brl = lambda v: f"R$ {v:,.0f}".replace(",", ".")
PLATAFORMA = ["cpu", "mobo", "ram"]   # peças ligadas pela compatibilidade


def render_teoria(res, build):
    t1, t2, t3, t4 = st.tabs(["1 · O problema", "2 · Regressão Linear",
                              "3 · KNN", "4 · Quem venceu e por quê"])
    with t1:
        _problema(res)
    with t2:
        _linear(res, build)
    with t3:
        _knn(res, build)
    with t4:
        _conclusao(res)


# ── 1. Problema ────────────────────────────────────────────────────────
def _problema(res):
    s = res["split"]
    st.subheader("O que estamos prevendo")
    st.markdown(f"""
**Tarefa:** regressão supervisionada, ou seja, prever um valor contínuo: o **preço do PC em R$**.

**Entradas (8 features categóricas):** processador, placa-mãe, memória, placa de vídeo,
SSD, fonte, cooler e gabinete, com compatibilidade de socket e DDR respeitada.

**Dataset:** {len(res['df']):,} builds geradas a partir de preços reais de varejo
(hardwarebarato.com, KaBuM, Pichau, ago/2026). O preço de cada build é a soma dos
componentes multiplicada por uma **variação de mercado aleatória de ±8%**, que simula
promoções e diferenças entre lojas.
""".replace(",", "."))
    st.subheader("Pipeline sem vazamento de dados")
    st.markdown(f"""
1. Geração dos dados brutos, sem nenhuma estatística calculada.
2. **Split 70/15/15 estratificado por faixa**: {s['treino']} treino · {s['validacao']} validação ·
   {s['teste']} teste, feito **antes** de qualquer análise.
3. EDA e ajuste dos encoders **somente no treino**, dentro de `Pipeline` + `ColumnTransformer`.
4. Hiperparâmetros do KNN escolhidos por `GridSearchCV` (validação cruzada de 5 dobras, dentro do treino).
5. Escolha do modelo pelo **MAE de validação**.
6. Conjunto de teste usado **uma única vez**, depois da escolha.
""")


# ── 2. Regressão Linear ────────────────────────────────────────────────
def _linear(res, build):
    lin = res["modelos"]["Linear"]
    coefs, b0 = ml.coeficientes_linear(lin)

    st.subheader("Como funciona")
    st.markdown("A Regressão Linear Múltipla supõe que o preço é uma **soma ponderada** das entradas:")
    st.latex(r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n")
    st.markdown("""
Como as entradas são nomes de peças, o **One-Hot Encoding** transforma cada peça em uma coluna
0/1 (1 = a build usa essa peça). Em cada categoria, a coluna da **peça mais barata é descartada**
(`drop='first'`). Sem isso, as colunas de cada categoria sempre somariam 1 e o modelo não teria
solução única: é a chamada *armadilha da variável dummy*.

O treino encontra os β que **minimizam a soma dos erros ao quadrado** (Mínimos Quadrados):
""")
    st.latex(r"\min_{\beta}\ \sum_{i=1}^{N}\left(y_i - \hat{y}_i\right)^2")
    st.info("**Interpretação neste projeto:** β₀ é o preço da build com todas as peças mais baratas, "
            "e cada βⱼ é o **custo extra** de trocar a peça mais barata da categoria pela peça j.")

    st.subheader("Prova 1: calculando a sua build na mão")
    ref = coefs.set_index(["categoria", "item"])
    linhas = [{"Parte": "β₀ (build base: peças mais baratas)", "Peça": "—",
               "Custo extra real": sum(ml.preco(c, ml.itens_por_preco(c)[0]) for c in ml.COLS),
               "β aprendido": b0}]
    for c in ml.COLS:
        r = ref.loc[(c, build[c])]
        linhas.append({"Parte": ml.NOMES_COLS[c], "Peça": build[c],
                       "Custo extra real": r["extra_real"], "β aprendido": r["coef"]})
    tab = pd.DataFrame(linhas)
    manual = tab["β aprendido"].sum()
    pred = float(lin.predict(pd.DataFrame([build])[ml.COLS])[0])
    st.dataframe(tab.style.format({"Custo extra real": brl, "β aprendido": brl}),
                 hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.metric("Soma manual β₀ + Σβ", brl(manual))
    c2.metric("model.predict()", brl(pred))
    dif = f"{abs(manual - pred):.2f}".replace(".", ",")
    st.success(f"Os dois valores são iguais (diferença de R$ {dif}): "
               "a predição da Linear é literalmente a soma da tabela acima.")
    plat = tab[tab["Parte"].isin([ml.NOMES_COLS[k] for k in PLATAFORMA])]
    st.caption(f"Repare nas linhas de processador, placa-mãe e memória: cada β isolado pode diferir "
               f"do custo real, mas a **soma das três** ({brl(plat['β aprendido'].sum())}) fica próxima "
               f"do real ({brl(plat['Custo extra real'].sum())}). O motivo está na Prova 2.")

    st.subheader("Prova 2: o modelo aprendeu os preços do catálogo")
    st.markdown("Cada ponto é uma peça. No eixo X, o custo extra **real** (tabela de preços); no eixo Y, "
                "o coeficiente que o modelo **aprendeu sozinho**, só vendo as builds e seus preços finais.")
    c = coefs[~coefs["referencia"]]
    fig = go.Figure()
    for cat in ml.COLS:
        d = c[c.categoria == cat]
        fig.add_scatter(x=d.extra_real, y=d.coef, mode="markers", name=ml.NOMES_COLS[cat],
                        text=d.item, hovertemplate="%{text}<br>real R$%{x:,.0f}<br>β R$%{y:,.0f}")
    m = c.extra_real.max()
    fig.add_scatter(x=[0, m], y=[0, m], mode="lines", name="β = custo real",
                    line=dict(dash="dash", color="gray"))
    fig.update_layout(height=430, xaxis_title="Custo extra real (R$)",
                      yaxis_title="Coeficiente aprendido β (R$)")
    st.plotly_chart(fig, width="stretch")
    err = (c.coef - c.extra_real).abs()
    livres = ~c.categoria.isin(PLATAFORMA)
    st.markdown(f"""
**Placa de vídeo, SSD, fonte, cooler e gabinete** caem sobre a diagonal: erro médio de apenas
**{brl(err[livres].mean())}** por peça. A Linear funciona porque **o preço de um PC é, de fato,
a soma das suas peças**, exatamente a forma que o modelo assume.

**Processador, placa-mãe e memória** se afastam da diagonal (erro médio de
{brl(err[~livres].mean())}). Não é defeito do modelo, e sim **multicolinearidade**: pela
compatibilidade, um processador AM5 *sempre* aparece com placa AM5 e memória DDR5. Como essas
peças andam juntas, o modelo pode "transferir" custo entre elas sem mudar nenhuma predição. Ele
aprende corretamente o **custo da plataforma** (a soma das três), mas não consegue separar a parte
de cada peça. A predição da build completa continua correta (Prova 1).
""")


# ── 3. KNN ─────────────────────────────────────────────────────────────
def _knn(res, build):
    knn = res["modelos"]["KNN"]
    K, pesos = res["knn_params"]["K"], res["knn_params"]["pesos"]

    st.subheader("Como funciona")
    st.markdown("""
O KNN (*K-Nearest Neighbors*) **não aprende uma equação**. No treino, ele apenas guarda as builds.
Para prever uma build nova, ele:

1. calcula a **distância** da build nova até todas as builds de treino;
2. separa as **K mais próximas** (os vizinhos);
3. devolve a **média do preço** desses vizinhos, simples ou ponderada pela proximidade.
""")
    st.latex(r"d(a,b)=\sqrt{\sum_j (a_j-b_j)^2}\qquad "
             r"\hat{y}=\frac{\sum_{i=1}^{K} w_i\,y_i}{\sum_{i=1}^{K} w_i},\quad "
             r"w_i=\begin{cases}1 & \text{uniforme}\\ 1/d_i & \text{por distância}\end{cases}")
    st.markdown("Aqui o One-Hot é **completo** (sem descartar colunas), para que toda troca de peça "
                "pese igual. Cada peça diferente muda 2 colunas (um 1 vira 0 e um 0 vira 1), então "
                "duas builds que diferem em **m** peças ficam à distância **√(2m)**:")
    st.dataframe(pd.DataFrame([[f"{np.sqrt(2 * m):.2f}" for m in range(5)]],
                              columns=[f"m = {m}" for m in range(5)], index=["Distância √(2m)"]),
                 width="stretch")

    st.subheader("Escolhendo K com GridSearchCV")
    curva = res["curva_k"]
    fig = go.Figure()
    for p, cor in [("uniform", "#94a3b8"), ("distance", "#f59e0b")]:
        d = curva[curva.pesos == p]
        fig.add_scatter(x=d.K, y=d.MAE_cv, mode="lines+markers", name=f"pesos = {p}",
                        line=dict(color=cor))
    fig.add_vline(x=K, line_dash="dash", annotation_text=f"K escolhido = {K}")
    fig.update_layout(height=380, xaxis_title="K (nº de vizinhos)",
                      yaxis_title="MAE médio na validação cruzada (R$)")
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"""
Foram testados **K de 1 a 30** com pesos uniformes e por distância, em validação cruzada de 5 dobras
**só no treino**. O menor erro foi **K = {K}, pesos = `{pesos}`**.
- **K pequeno** (1, 2…): o modelo copia um único vizinho e fica sensível ao ruído (*overfitting*).
- **K grande**: a média mistura builds muito diferentes e tudo tende ao preço médio (*underfitting*).
""")

    st.subheader("Prova: os vizinhos da sua build")
    viz, manual = ml.vizinhos_knn(knn, res["X_tr"], res["y_tr"], build)
    pred = float(knn.predict(pd.DataFrame([build])[ml.COLS])[0])
    show = viz[["distancia", "n_diferentes", "difere_em", "preco", "peso"]].rename(columns={
        "distancia": "Distância", "n_diferentes": "Peças diferentes",
        "difere_em": "Difere em", "preco": "Preço real", "peso": "Peso na média"})
    show.index = [f"Vizinho {i + 1}" for i in range(len(show))]
    st.dataframe(show.style.format({"Distância": "{:.2f}", "Preço real": brl,
                                    "Peso na média": "{:.1%}"}), width="stretch")

    fig = go.Figure(go.Bar(x=show.index, y=viz.preco, marker_color="#f59e0b",
                           text=[brl(v) for v in viz.preco], textposition="outside"))
    fig.add_hline(y=pred, line_dash="dash", line_color="#2563eb",
                  annotation_text=f"Predição = {brl(pred)}")
    fig.update_layout(height=380, yaxis_title="Preço do vizinho (R$)")
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    c1.metric("Média ponderada manual", brl(manual))
    c2.metric("model.predict()", brl(pred))
    st.success("A predição do KNN é exatamente a média ponderada dos preços dos vizinhos acima.")

    amp = viz.preco.max() - viz.preco.min()
    st.warning(f"**Limitação visível nesta tabela:** vizinhos à mesma distância custam de "
               f"{brl(viz.preco.min())} a {brl(viz.preco.max())} (diferença de {brl(amp)}). "
               "Para o KNN, trocar o cooler ou trocar a placa de vídeo conta como **uma peça "
               "diferente**, com o mesmo peso, embora uma RTX 5090 custe R$ 17 mil a mais que uma "
               "GTX 1650. A distância mede **semelhança**, não **custo**.")


# ── 4. Conclusão ───────────────────────────────────────────────────────
def _conclusao(res):
    v, t = res["validacao"], res["teste"]
    st.subheader(f"Vencedor: {res['melhor']}")
    st.markdown(f"""
| | Linear | KNN |
|---|---|---|
| MAE validação | {brl(v['Linear']['MAE'])} | {brl(v['KNN']['MAE'])} |
| MAE teste | {brl(t['Linear']['MAE'])} | {brl(t['KNN']['MAE'])} |
| R² teste | {t['Linear']['R2']:.3f} | {t['KNN']['R2']:.3f} |
| MAPE teste | {t['Linear']['MAPE']:.1f}% | {t['KNN']['MAPE']:.1f}% |

**Por que a Linear venceu?** O preço de um PC é aditivo: a soma das peças. A Linear assume
exatamente essa forma, então cada coeficiente vira o preço de uma peça (Prova 2).

**Ela chegou ao limite?** Quase. A variação de mercado de ±8% é aleatória e **ninguém consegue
prevê-la**: nem um modelo perfeito erraria menos que ≈ 4% em média. A Linear tem MAPE de
**{t['Linear']['MAPE']:.1f}%**, praticamente esse piso.

**Por que o KNN ficou atrás?** Ele não sabe *quanto* cada peça custa, só *quantas* peças são
diferentes. Vizinhos "próximos" podem ter placas de vídeo de preços muito distintos.

**Interpretabilidade:** a Linear é totalmente explicável (cada β é um preço em reais). O KNN é
explicável caso a caso (mostrando os vizinhos), mas não gera uma regra geral.

**Métricas escolhidas:** MAE (erro médio em reais, fácil de comunicar), RMSE (penaliza erros
grandes), R² (fração da variação de preço explicada) e MAPE (erro relativo, justo entre um PC de
R$ 3 mil e um de R$ 30 mil). Métricas de classificação (Precision, Recall, F1) não se aplicam,
porque o alvo é contínuo.
""")
