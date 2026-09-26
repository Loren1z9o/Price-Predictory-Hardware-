// Relatório ABNT — PC Gamer Price Predictor v7.0 (segue o modelo "Projeto Final" do professor)
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, Header,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, PageNumber, TableOfContents,
  PageBreak, LineRuleType, ShadingType,
} = require("docx");

const d = JSON.parse(fs.readFileSync("report_data.json", "utf8"));
const R = (n) => "R$ " + Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
const N = (n, c = 1) => n.toFixed(c).replace(".", ",");
const P100 = (n, c = 1) => N(n * 100, c) + "%";
const L = d.teste.Linear, K = d.teste.KNN, VL = d.validacao.Linear, VK = d.validacao.KNN;
const kp = d.knn_params;

const FONT = "Times New Roman";
const LINE15 = { line: 360, lineRule: LineRuleType.AUTO };
const TEXT_W = 9071; // 16 cm em DXA

// ── helpers ─────────────────────────────────────────────────────────────
function runs(text, base = {}) {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*|_[^_]+_)/).filter(Boolean).map((t) => {
    if (t.startsWith("`")) return new TextRun({ text: t.slice(1, -1), ...base, font: "Courier New", size: (base.size || 24) - 2 });
    if (t.startsWith("**")) return new TextRun({ text: t.slice(2, -2), bold: true, ...base });
    if (t.startsWith("_") && t.endsWith("_")) return new TextRun({ text: t.slice(1, -1), italics: true, ...base });
    return new TextRun({ text: t, ...base });
  });
}
const P = (t) => new Paragraph({ children: runs(t), alignment: AlignmentType.JUSTIFIED,
  spacing: LINE15, indent: { firstLine: 709 } });
const Pc = (t, o = {}) => new Paragraph({ children: runs(t, o), alignment: AlignmentType.CENTER, spacing: LINE15 });
const AL = (t) => new Paragraph({ children: runs(t), alignment: AlignmentType.JUSTIFIED,
  spacing: LINE15, indent: { left: 709, hanging: 425 } });
const EQ = (t, n) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120, ...LINE15 },
  children: [new TextRun({ text: t, italics: true }), new TextRun({ text: `\t\t(${n})` })] });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: true });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const blank = (n = 1) => Array.from({ length: n }, () => new Paragraph({ spacing: LINE15, children: [] }));
const small = (t, align = AlignmentType.LEFT) => new Paragraph({ alignment: align,
  spacing: { before: 40, after: 200 }, children: runs(t, { size: 20 }) });
const FONTE = "Fonte: elaborado pelos autores (2026).";

let nFig = 0, nTab = 0;
function figura(file, legenda, wIn, hIn) {
  nFig++;
  const w = 600, h = Math.round(w * hIn / wIn);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 60 }, keepNext: true,
      children: runs(`Figura ${nFig} – ${legenda}`, { size: 20 }) }),
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: w, height: h } })] }),
    small(FONTE, AlignmentType.CENTER),
  ];
}
function tabela(legenda, head, rows, widths) {
  nTab++;
  const total = widths.reduce((a, b) => a + b, 0);
  const W = widths.map((x) => Math.round(x * TEXT_W / total));
  W[W.length - 1] += TEXT_W - W.reduce((a, b) => a + b, 0);
  const line = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const cell = (t, i, isHead, isLast) => new TableCell({
    width: { size: W[i], type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    shading: isHead ? { type: ShadingType.CLEAR, fill: "EDEDED", color: "auto" } : undefined,
    borders: { top: isHead ? line : none, bottom: isHead || isLast ? line : none, left: none, right: none },
    children: [new Paragraph({ keepNext: true, alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(t), { size: 20, bold: isHead }) })],
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 60 }, keepNext: true,
      children: runs(`Tabela ${nTab} – ${legenda}`, { size: 20 }) }),
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: W,
      rows: [new TableRow({ tableHeader: true, cantSplit: true, children: head.map((t, i) => cell(t, i, true, false)) }),
        ...rows.map((r, j) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, j === rows.length - 1)) }))] }),
    small(FONTE, AlignmentType.LEFT),
  ];
}

// ── pré-textuais ────────────────────────────────────────────────────────
const TITULO = "ESTIMATIVA DO PREÇO DE COMPUTADORES GAMER POR REGRESSÃO LINEAR MÚLTIPLA E K-VIZINHOS MAIS PRÓXIMOS";
const AUTORES = ["LORENZO SARTORI", "JOÃO GUSTAVO", "LUCAS RODRIGUES VIGARANI"];

const capa = [
  Pc("**CENTRO UNIVERSITÁRIO UNISATC**"), Pc("**CURSO DE ENGENHARIA DE COMPUTAÇÃO**"),
  ...blank(4), ...AUTORES.map((a) => Pc(a)), ...blank(6), Pc(`**${TITULO}**`), ...blank(10),
  Pc("CRICIÚMA/SC"), Pc("2026"),
];
const rosto = [
  new Paragraph({ children: [new PageBreak()] }),
  ...AUTORES.map((a) => Pc(a)), ...blank(6), Pc(`**${TITULO}**`), ...blank(3),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 4536 },
    children: runs("Relatório do Projeto Final apresentado à disciplina de Machine Learning Clássico do Curso de Engenharia de Computação do Centro Universitário UniSATC.", { size: 20 }) }),
  ...blank(1),
  new Paragraph({ indent: { left: 4536 }, children: runs("Professor: Prof. Dr. Rodrigo Ramos Silva", { size: 20 }) }),
  ...blank(8), Pc("CRICIÚMA/SC"), Pc("2026"),
];
const resumo = [
  new Paragraph({ children: [new PageBreak()] }), Pc("**RESUMO**"), ...blank(1),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { line: 240 }, children: runs(
    `Este trabalho desenvolve e compara dois modelos de Machine Learning Clássico, a Regressão Linear Múltipla e o K-Vizinhos Mais Próximos (KNN), para estimar o preço, em reais, de computadores gamer a partir de oito componentes: processador, placa-mãe, memória, placa de vídeo, SSD, fonte, cooler e gabinete. O conjunto de dados contém ${d.n_total.toLocaleString("pt-BR")} montagens compatíveis geradas a partir de preços reais do varejo brasileiro (agosto de 2026), com variação de mercado de ±8%. O protocolo seguiu uma ordem de prevenção de vazamento de dados: divisão 70/15/15 estratificada antes de qualquer análise, codificação One-Hot dentro de pipelines ajustados apenas no treino, otimização do KNN por GridSearchCV, seleção pela validação e uso único do conjunto de teste. A Regressão Linear venceu, com MAE de ${R(L.MAE)}, R² de ${N(L.R2, 3)} e MAPE de ${N(L.MAPE)}% no teste, contra ${R(K.MAE)}, ${N(K.R2, 3)} e ${N(K.MAPE)}% do KNN (K = ${kp.K}). Mostra-se que os coeficientes aprendidos pela Linear recuperam o preço de cada peça do catálogo e que seu erro está próximo do piso de 4% imposto pela variação de mercado, enquanto o KNN é limitado por medir semelhança entre montagens, e não custo.`, { size: 24 })}),
  ...blank(1),
  new Paragraph({ children: runs("**Palavras-chave:** regressão linear; KNN; precificação de hardware; aprendizado supervisionado; One-Hot Encoding.") }),
];
const sumario = [
  new Paragraph({ children: [new PageBreak()] }), Pc("**SUMÁRIO**"),
  new TableOfContents("Sumário", { hyperlink: true, headingStyleRange: "1-3" }),
];

// ── 1 INTRODUÇÃO ────────────────────────────────────────────────────────
const s1 = [
  H1("1 INTRODUÇÃO, PROBLEMA E JUSTIFICATIVA"),
  H2("1.1 Contextualização do problema real"),
  P("O domínio deste trabalho é o **varejo de hardware para computadores de jogos no Brasil**. O computador é uma plataforma relevante de entretenimento no país: segundo a Pesquisa Game Brasil 2026, que entrevistou 7.115 pessoas, 75,3% dos brasileiros se consideram gamers e o computador é a plataforma preferida de 21,1% deles, com tendência de alta (HILDEBRAND, 2026). No mundo, o segmento de PC deve movimentar US$ 39,9 bilhões em 2025, 21% da receita global de jogos, com 936 milhões de jogadores (MEIO & MENSAGEM, 2025)."),
  P("Montar um computador gamer exige escolher, entre centenas de opções, peças compatíveis entre si. Hoje, quem quer saber quanto custará uma configuração soma manualmente os preços consultados em lojas ou em comparadores, como o hardwarebarato.com, que agrega ofertas de dezenas de varejistas (HARDWARE BARATO, 2026). Esse processo é lento, sujeito a erro e precisa ser refeito a cada combinação avaliada, o que dificulta comparar alternativas e identificar ofertas fora do padrão."),
  P("A estimativa do preço de um computador a partir das suas características é um problema reconhecido na literatura, sob o nome de **regressão hedônica**: o produto é tratado como um conjunto de atributos, e o preço, como função deles (ROSEN, 1974). Berndt, Griliches e Rappaport (1995) aplicaram essa abordagem a computadores pessoais nos Estados Unidos e, no Brasil, Fouto, De Angelo e Luppe (2009) identificaram, em 3.779 desktops anunciados entre 2003 e 2007, os atributos que mais explicam o preço, entre eles memória, processador e memória de vídeo."),
  H2("1.2 Definição do problema de Machine Learning"),
  P("**Definição técnica.** Trata-se de um problema de **regressão supervisionada**: dado um conjunto de componentes, prever um valor contínuo."),
  P("**Variável-alvo.** A coluna `preco`, que é o preço total da montagem (_build_) em reais."),
  P(`**Conjunto de dados.** Não existe um conjunto público com preços de montagens brasileiras descritas peça a peça. Por isso, o conjunto foi **gerado a partir de um catálogo com preços reais** coletados no hardwarebarato.com e em grandes varejistas (KaBuM, Pichau) em agosto de 2026. O catálogo tem ${Object.values(d.n_itens).reduce((a, b) => a + b)} peças em oito categorias e regras de compatibilidade de soquete (AM4, AM5, LGA1700) e de geração de memória (DDR4, DDR5). Foram simuladas ${d.n_total.toLocaleString("pt-BR")} montagens em cinco faixas (entry a enthusiast). O preço de cada uma é a soma das peças multiplicada por uma **variação de mercado aleatória de ±8%**, que representa promoções e diferenças entre lojas. O gerador está no código-fonte (\`gerar_dataset\`, semente 42), o que garante reprodutibilidade total.`),
  P("Esse conjunto é adequado porque usa preços reais, respeita a compatibilidade entre peças e tem uma **estrutura verdadeira conhecida**. Isso permite verificar não só _quanto_ cada modelo erra, mas _se_ ele aprendeu a relação correta, o que não é possível com dados de mercado. A limitação correspondente, discutida na Seção 4.4, é que preços reais contêm efeitos não aditivos ausentes na simulação."),
  H2("1.3 Justificativa e relevância"),
  P("**Impacto.** Uma estimativa instantânea do preço de uma configuração permite ao consumidor comparar alternativas e planejar a compra, e ao lojista precificar montagens sob encomenda. Um modelo interpretável ainda mostra **quanto cada peça contribui** para o total, informação útil para decidir onde economizar."),
  P("**Relevância do Machine Learning.** Mesmo com o catálogo reduzido deste trabalho, existem milhões de combinações possíveis de peças, impossíveis de tabelar. Um modelo supervisionado aprende o valor de cada peça a partir de exemplos de preços **totais**, sem que ninguém informe o preço individual, e generaliza para combinações nunca vistas. Isso substitui a soma manual por uma estimativa automática e atualizável: basta treinar novamente com preços novos."),
];

// ── 2 OBJETIVOS ─────────────────────────────────────────────────────────
const s2 = [
  H1("2 OBJETIVOS"),
  H2("2.1 Objetivo geral"),
  P("Desenvolver e comparar dois modelos de Machine Learning Clássico, a Regressão Linear Múltipla e o K-Vizinhos Mais Próximos (KNN), para estimar o preço em reais de computadores gamer a partir dos componentes escolhidos, explicando como cada modelo chega à sua estimativa."),
  H2("2.2 Objetivos específicos"),
  AL("a)\tobter e estruturar um conjunto de dados de montagens a partir de um catálogo de peças com preços reais do mercado brasileiro e regras de compatibilidade;"),
  AL("b)\trealizar a análise exploratória de dados (EDA), somente no conjunto de treino, para diagnosticar a qualidade dos dados;"),
  AL("c)\timplementar o pipeline de pré-processamento e engenharia de atributos sem vazamento de dados;"),
  AL("d)\ttreinar e otimizar dois modelos, a Regressão Linear Múltipla e o KNN, este com busca de hiperparâmetros por validação cruzada;"),
  AL("e)\tcomparar o desempenho dos modelos com métricas de regressão, na validação e, uma única vez, no teste;"),
  AL("f)\tanalisar a contribuição de cada atributo por meio dos coeficientes da Regressão Linear e dos vizinhos do KNN;"),
  AL("g)\tdesenvolver um protótipo de _deployment_ em Streamlit que estime o preço e demonstre, passo a passo, o funcionamento dos dois modelos."),
];

// ── 3 FUNDAMENTAÇÃO ─────────────────────────────────────────────────────
const s3 = [
  H1("3 FUNDAMENTAÇÃO TEÓRICA"),
  H2("3.1 Revisão bibliográfica do domínio"),
  H3("3.1.1 Conceitos-chave do domínio"),
  P("Uma **build** é a combinação de peças que compõe um computador montado. Neste trabalho, ela é descrita por oito componentes: **processador** (CPU), responsável pelo processamento geral; **placa-mãe**, que interliga as peças e define o soquete e o tipo de memória suportados; **memória RAM**; **placa de vídeo** (GPU), responsável pela renderização dos jogos; **SSD**, para armazenamento; **fonte**, que fornece energia; **cooler**, que refrigera o processador; e **gabinete**."),
  P("A **compatibilidade** restringe as combinações possíveis: o processador precisa ter o mesmo **soquete** da placa-mãe (AM4, AM5 ou LGA1700), e a memória precisa ser da **geração** suportada pela placa (DDR4 ou DDR5). Por fim, o **preço hedônico** é o valor implícito de cada atributo de um produto, estimado a partir dos preços totais observados (ROSEN, 1974). Esse é exatamente o papel que os coeficientes da Regressão Linear assumem neste trabalho."),
  H3("3.1.2 Estatísticas e fatos"),
  P(`Além dos números de mercado da Seção 1.1, o próprio catálogo, calibrado em preços reais, evidencia a estrutura do problema. No conjunto de treino, a placa de vídeo responde, em média, por ${P100(d.gpu_share_media, 0)} do preço de tabela de uma montagem, e por ${P100(d.gpu_share_enth, 0)} nas montagens enthusiast. O preço das montagens vai de ${R(d.eda.preco.min)} a ${R(d.eda.preco.max)}, com mediana de ${R(d.eda.preco["50%"])}. Somente a diferença entre a placa de vídeo mais barata (GTX 1650) e a mais cara (RTX 5090) é de ${R(d.gpu_5090[0])}.`),
  H3("3.1.3 Soluções não baseadas em Machine Learning"),
  P("As soluções usuais são: (a) a **soma manual** de preços em planilhas; (b) os **comparadores de preço**, que agregam ofertas de várias lojas, mas exigem consulta item a item; (c) os **configuradores** das lojas (\"monte seu PC\"), limitados ao estoque de uma única loja; e (d) a **regressão hedônica** estatística, que exige um especialista para especificar o modelo. As três primeiras não generalizam: dependem de consulta atualizada para cada peça e não estimam configurações ou preços não pesquisados. Nenhuma delas aprende automaticamente com novos dados, lacuna que motiva a abordagem por Machine Learning."),
  H3("3.1.4 Trabalhos relacionados"),
  P("Com o aprendizado de máquina, a precificação de computadores passou a ser tratada como regressão supervisionada, em geral para notebooks. Siburian et al. (2022) compararam métodos de _ensemble_ e obtiveram o melhor resultado com o XGBoost (R² de 92,77%). Tian (2024), com 1.320 notebooks e 13 atributos, obteve R² de 0,62 com Regressão Linear e de 0,85 com XGBoost. Nesses estudos, com preços reais, métodos não lineares superam a Regressão Linear, ponto retomado na análise crítica."),
  H2("3.2 Referencial teórico em Machine Learning"),
  H3("3.2.1 Modelo A: Regressão Linear Múltipla"),
  P("A Regressão Linear Múltipla é um modelo **paramétrico** para problemas de **regressão**. Ela supõe que o alvo é uma combinação linear das entradas (JAMES et al., 2013):"),
  EQ("ŷ = β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ", 1),
  P("em que β₀ é o intercepto e cada βⱼ é o efeito de uma unidade de xⱼ sobre o preço, mantidas as demais constantes. O treino, pelo método dos **Mínimos Quadrados Ordinários**, escolhe os coeficientes que minimizam a soma dos erros ao quadrado:"),
  EQ("min Σᵢ (yᵢ − ŷᵢ)²", 2),
  P("Quando as entradas são categóricas, cada categoria vira uma variável binária (_dummy_). Se todas as categorias de um atributo forem mantidas junto com o intercepto, as colunas somam 1 em toda linha e a matriz perde posto, a chamada **armadilha da variável _dummy_**. Por isso, descarta-se uma categoria de referência, e cada coeficiente passa a medir a diferença em relação a ela (HASTIE; TIBSHIRANI; FRIEDMAN, 2009). Quando atributos diferentes andam sempre juntos, ocorre **multicolinearidade**: as predições continuam corretas, mas os coeficientes individuais deixam de ser identificáveis."),
  P("Suas principais vantagens são a **interpretabilidade**, pois cada coeficiente tem significado direto, e a ausência de hiperparâmetros. A limitação é supor uma relação aditiva e linear, o que pode falhar em problemas com interações entre atributos."),
  H3("3.2.2 Modelo B: K-Vizinhos Mais Próximos (KNN)"),
  P("O KNN é um método **não paramétrico** e **baseado em instâncias**, aplicável a **classificação e regressão** (COVER; HART, 1967). Ele não estima uma equação: no treino, apenas armazena os exemplos. Para prever um novo ponto, calcula sua distância até todos os exemplos de treino, seleciona os K mais próximos e, na regressão, devolve a média dos seus alvos, simples ou ponderada pelo inverso da distância:"),
  EQ("d(a, b) = √Σⱼ (aⱼ − bⱼ)²", 3),
  EQ("ŷ = Σᵢ wᵢ yᵢ / Σᵢ wᵢ,   com wᵢ = 1 (uniforme) ou wᵢ = 1/dᵢ (por distância)", 4),
  P("O hiperparâmetro K controla o compromisso entre viés e variância: valores pequenos copiam poucos vizinhos e ficam sensíveis a ruído (_overfitting_), e valores grandes fazem a média abranger exemplos pouco parecidos (_underfitting_) (JAMES et al., 2013). Como o KNN depende de distâncias, a **representação e a escala das variáveis** determinam o que ele considera \"parecido\". Ele é flexível e não supõe forma para a relação, mas não produz uma regra geral e seu custo de predição cresce com o tamanho do treino."),
  H3("3.2.3 Modelos de ensemble"),
  P("Não foram utilizados. A disciplina exige ao menos dois modelos, e optou-se por dois algoritmos de naturezas opostas, um paramétrico e outro baseado em instâncias, cujo funcionamento pode ser demonstrado passo a passo. Isso privilegia a interpretabilidade, um dos focos deste trabalho."),
];

// ── 4 METODOLOGIA ───────────────────────────────────────────────────────
const catRows = Object.entries(d.n_itens).map(([k, v]) => [
  { cpu: "Processador", mobo: "Placa-mãe", ram: "Memória RAM", gpu: "Placa de vídeo", ssd: "SSD", fonte: "Fonte", cooler: "Cooler", gabinete: "Gabinete" }[k],
  k, "Categórica nominal", v]);
const e = d.eda;
const linRows = d.ex_lin_rows.map((r) => [r[0], r[1], R(r[2]), R(r[3])]);
const somaB = d.b0 + d.ex_lin_rows.reduce((a, r) => a + r[3], 0);
const plat = d.ex_lin_rows.filter((r) => ["Processador", "Placa-mãe", "Memória RAM"].includes(r[0]));
const knnRows = d.ex_knn_rows.map((r, i) => [`Vizinho ${i + 1}`, N(r[0], 2), r[1], r[2], R(r[3]), P100(r[4])]);
const precosViz = d.ex_knn_rows.map((r) => r[3]);

const s4 = [
  H1("4 METODOLOGIA E DESENVOLVIMENTO"),
  P("O pipeline seguiu a ordem de prevenção de vazamento de dados recomendada na disciplina (KAUFMAN et al., 2012): geração dos dados brutos, **divisão treino/validação/teste antes de qualquer estatística**, EDA somente no treino, pré-processamento encapsulado em _pipelines_ ajustados apenas no treino, otimização por validação cruzada no treino, seleção do modelo pela validação e **uso único do conjunto de teste**."),
  H2("4.1 Análise exploratória de dados (EDA)"),
  P(`**Descrição.** O conjunto tem ${d.n_total.toLocaleString("pt-BR")} linhas e 10 colunas: as oito variáveis explicativas categóricas (Tabela 1), o alvo \`preco\` (numérico contínuo) e a coluna auxiliar \`tier\` (faixa da montagem). Esta última é usada somente para estratificar a divisão e **não entra nos modelos**, pois o usuário não a informa. A EDA foi feita apenas nas ${e.linhas_treino.toLocaleString("pt-BR")} linhas de treino.`),
  ...tabela("Variáveis explicativas do conjunto de dados", ["Variável", "Coluna", "Tipo", "Nº de categorias"], catRows, [3, 2, 3, 2]),
  P(`**Diagnóstico.** Não há **valores ausentes** (${e.nulos} nulos). O preço no treino tem média de ${R(e.preco.mean)}, mediana de ${R(e.preco["50%"])} e desvio-padrão de ${R(e.preco.std)}, com assimetria à direita (média maior que a mediana). O critério do intervalo interquartil (IQR) aponta **${e.outliers_iqr} outliers** acima de ${R(e.limite_superior_iqr)}. Eles **não foram removidos**, pois são montagens enthusiast legítimas, como as equipadas com RTX 5090, que o modelo precisa aprender a precificar. Como o problema é de regressão, não há desbalanceamento de classes; a distribuição das faixas no treino é: entry ${e.builds_por_tier.entry}, budget ${e.builds_por_tier.budget}, mid ${e.builds_por_tier.mid}, high ${e.builds_por_tier.high} e enthusiast ${e.builds_por_tier.enthusiast}, preservada nos três conjuntos pela estratificação.`),
  P("**Visualizações.** A Figura 1 mostra o histograma do preço por faixa, com o limite superior do IQR."),
  ...figura("fig/f1_eda.png", "Distribuição do preço das montagens no treino, por faixa", 6.5, 3.2),
  H2("4.2 Pré-processamento e engenharia de atributos"),
  P(`**a) Divisão dos dados.** Proporção **70/15/15** (${d.split.treino} treino, ${d.split.validacao} validação, ${d.split.teste} teste), estratificada por faixa, com semente fixa, executada **antes de qualquer análise**. O treino ajusta os modelos; a validação escolhe entre eles; o teste, usado uma única vez, estima o desempenho em dados nunca vistos. Separar a validação do teste impede que a escolha do modelo contamine a estimativa final.`),
  P("**b) Tratamento de nulos.** Não foi necessário, pois não há valores ausentes. Por isso, nenhum imputador foi incluído no _pipeline_: adicioná-lo seria uma etapa sem efeito."),
  P(`**c) Codificação categórica.** Aplicou-se **One-Hot Encoding**, adequado a variáveis nominais sem ordem natural. Os dois modelos usam variações diferentes, cada uma justificada pelo algoritmo. Na **Linear**, usa-se \`drop='first'\` com as categorias ordenadas por preço, de modo que a coluna descartada em cada atributo é sempre a **peça mais barata**. Isso evita a armadilha da _dummy_ e faz de cada coeficiente o custo extra de trocar a peça mais barata por aquela (${d.n_onehot - 8} colunas). No **KNN**, usa-se One-Hot **completo** (${d.n_onehot} colunas), para que toda troca de peça pese igual na distância: cada peça diferente altera duas colunas, e duas montagens que diferem em m peças ficam à distância **√(2m)**. O \`drop='first'\` distorceria essa distância, pois trocas envolvendo a peça de referência alterariam apenas uma coluna.`),
  P("**d) Escalonamento.** **Não foi aplicado**, e a decisão é justificada. Todas as entradas são binárias (0 ou 1) e, portanto, já estão na mesma escala. A Regressão Linear é invariante à escala nas predições, e o escalonamento apenas mudaria a unidade dos coeficientes, que deixariam de estar em reais. No KNN, o StandardScaler daria peso maior às peças raras (colunas com menor desvio-padrão), distorcendo a noção de semelhança."),
  P("**e) Engenharia de atributos.** A representação One-Hot ordenada por preço é o principal atributo construído, pois torna os coeficientes diretamente interpretáveis. A compatibilidade de soquete e memória foi codificada na geração dos dados e na interface. Especificações numéricas (VRAM, capacidade em GB, potência) foram avaliadas e **descartadas**: são funções determinísticas da peça e, junto com o One-Hot, gerariam multicolinearidade perfeita na Linear. A pertinência desses atributos para o KNN é discutida na Seção 4.4."),
  H2("4.3 Implementação, treinamento e otimização dos modelos"),
  H3("4.3.1 Detalhes de implementação"),
  P("Os modelos foram implementados em Python com a biblioteca **scikit-learn** (PEDREGOSA et al., 2011). Cada modelo é um `Pipeline` composto por um `ColumnTransformer` com o `OneHotEncoder` e pelo estimador (`LinearRegression` ou `KNeighborsRegressor`). Como o codificador está dentro do _pipeline_, ele é ajustado apenas nos dados de treino de cada etapa, inclusive em cada dobra da validação cruzada, o que elimina o vazamento."),
  H3("4.3.2 Otimização"),
  P(`A **Regressão Linear** não tem hiperparâmetros a ajustar: os coeficientes são obtidos pela solução de Mínimos Quadrados. Para comparação, seu erro foi estimado por validação cruzada de 5 dobras no treino (MAE de ${R(d.cv_mae.Linear)}).`),
  P(`O **KNN** foi otimizado com **GridSearchCV**, em validação cruzada de 5 dobras embaralhadas **somente no treino** (KOHAVI, 1995), minimizando o MAE. A grade combinou K de 1 a 30 com pesos uniformes e por distância (60 combinações, 300 ajustes). A métrica de distância não entrou na grade: para vetores binários, a distância de Manhattan (2m) e a euclidiana (√(2m)) ordenam os vizinhos da mesma forma. A melhor configuração foi **K = ${kp.K} com pesos por distância** (MAE de ${R(d.cv_mae.KNN)} na validação cruzada). A Figura 2 mostra a curva: com K pequeno o erro é alto pela sensibilidade ao ruído, e a partir de K ≈ ${kp.K} a média passa a incluir montagens pouco parecidas.`),
  ...figura("fig/f3_k.png", "MAE da validação cruzada em função de K no KNN", 6.5, 3.0),
  H3("4.3.3 Funcionamento dos modelos em um exemplo resolvido"),
  P(`Para demonstrar como cada modelo chega à estimativa, considera-se a montagem: ${d.ex_build.cpu}, ${d.ex_build.mobo}, ${d.ex_build.ram}, ${d.ex_build.gpu}, ${d.ex_build.ssd}, ${d.ex_build.fonte}, ${d.ex_build.cooler} e ${d.ex_build.gabinete}, cujo preço de tabela é ${R(d.ex_tabela)}.`),
  P(`**Regressão Linear.** A predição é a soma do intercepto com os coeficientes das peças escolhidas (Tabela 2). O intercepto aprendido, ${R(d.b0)}, corresponde à montagem com todas as peças mais baratas, cujo valor real é ${R(d.b0_real)}.`),
  ...tabela("Predição da Regressão Linear decomposta por peça",
    ["Parte", "Peça", "Custo extra real", "β aprendido"],
    [["β₀ (montagem base)", "peças mais baratas", R(d.b0_real), R(d.b0)], ...linRows,
     ["**Total**", "", `**${R(d.ex_tabela)}**`, `**${R(somaB)}**`]], [2.4, 3.6, 1.8, 1.8]),
  P(`A soma manual, ${R(somaB)}, é idêntica ao resultado de \`model.predict()\`. Os coeficientes de placa de vídeo, SSD, fonte, cooler e gabinete coincidem com os custos reais. Nas linhas de processador, placa-mãe e memória, os valores individuais diferem, mas a soma das três (${R(plat.reduce((a, r) => a + r[3], 0))}) é próxima da real (${R(plat.reduce((a, r) => a + r[2], 0))}). Isso é multicolinearidade: pela compatibilidade, essas peças aparecem sempre juntas, e o modelo aprende o custo da **plataforma**, não o de cada peça.`),
  P(`**KNN.** O modelo localiza as ${kp.K} montagens de treino mais próximas (Tabela 3). A predição é a média dos seus preços ponderada pelo inverso da distância, ${R(d.ex_knn_manual)}, idêntica ao resultado de \`model.predict()\`.`),
  ...tabela(`Os ${kp.K} vizinhos mais próximos da montagem de exemplo no KNN`,
    ["Vizinho", "Distância", "Peças diferentes", "Difere em", "Preço real", "Peso"], knnRows, [1.5, 1.2, 1.3, 4.2, 1.6, 1.0]),
  P(`A tabela evidencia a limitação do KNN neste problema: vizinhos à mesma distância custam de ${R(Math.min(...precosViz))} a ${R(Math.max(...precosViz))}. Para o algoritmo, trocar o cooler ou trocar a placa de vídeo conta igualmente como uma peça diferente, embora o impacto no preço seja muito distinto. A distância mede **semelhança**, não **custo**.`),
  H2("4.4 Avaliação e análise de resultados"),
  H3("4.4.1 Comparação quantitativa"),
  P(`A Tabela 4 compara os modelos. A escolha do vencedor foi feita pelo **MAE de validação**; o conjunto de teste foi avaliado uma única vez, depois da escolha, apenas para estimar o desempenho final.`),
  ...tabela("Métricas dos modelos na validação e no teste",
    ["Métrica", "Linear (validação)", "KNN (validação)", "Linear (teste)", "KNN (teste)"],
    [["MAE", R(VL.MAE), R(VK.MAE), `**${R(L.MAE)}**`, R(K.MAE)],
     ["RMSE", R(VL.RMSE), R(VK.RMSE), `**${R(L.RMSE)}**`, R(K.RMSE)],
     ["R²", N(VL.R2, 3), N(VK.R2, 3), `**${N(L.R2, 3)}**`, N(K.R2, 3)],
     ["MAPE", N(VL.MAPE) + "%", N(VK.MAPE) + "%", `**${N(L.MAPE)}%**`, N(K.MAPE) + "%"]], [1.4, 2, 2, 2, 2]),
  H3("4.4.2 Justificativa das métricas"),
  P("Como o alvo é contínuo, as métricas de classificação sugeridas no modelo de relatório (_Precision_, _Recall_, F1, acurácia, matriz de confusão e curva ROC) **não se aplicam**. Foram usadas as métricas de regressão da disciplina. O **MAE** é a métrica principal: expressa o erro médio em reais e é fácil de comunicar ao usuário. O **RMSE** penaliza mais os erros grandes e, comparado ao MAE, revela se há erros extremos. O **R²** indica a fração da variação dos preços explicada pelo modelo. O **MAPE** mede o erro relativo, que é mais justo entre uma montagem de R$ 3 mil e uma de R$ 30 mil."),
  H3("4.4.3 Visualizações"),
  P("A Figura 3 compara preço real e previsto no teste. Os pontos da Linear se concentram sobre a diagonal, enquanto os do KNN se dispersam e ficam abaixo dela nas montagens mais caras: a média de vizinhos \"puxa\" as estimativas para o centro. A Figura 4 mostra os resíduos. Os da Linear são estreitos e centrados em zero; os do KNN são largos e assimétricos. A Figura 5 é a principal evidência do funcionamento da Linear: os coeficientes aprendidos, obtidos apenas a partir dos preços totais, coincidem com os custos reais das peças."),
  ...figura("fig/f4_pred.png", "Preço real × previsto no conjunto de teste", 6.5, 3.1),
  ...figura("fig/f5_res.png", "Distribuição dos resíduos no conjunto de teste", 6.5, 2.9),
  ...figura("fig/f2_coef.png", "Coeficientes aprendidos pela Regressão Linear × custo extra real das peças", 6.5, 3.6),
  H3("4.4.4 Contribuição dos atributos"),
  P(`Na Regressão Linear, a contribuição de cada atributo é lida diretamente nos coeficientes, em reais. A placa de vídeo domina: seus coeficientes vão de zero a ${R(d.gpu_5090[1])} (RTX 5090, contra ${R(d.gpu_5090[0])} reais), enquanto os do cooler ou do gabinete não passam de R$ 1.100. A correlação entre coeficientes e custos reais é de ${N(d.coef_corr, 3)}. No KNN não há importância global de atributos; a explicação é local, pelos vizinhos de cada predição (Tabela 3).`),
  H3("4.4.5 Análise crítica"),
  P(`**A Regressão Linear foi o melhor modelo**, com erro cerca de ${N(K.MAE / L.MAE, 1)} vezes menor que o do KNN. A razão é estrutural: o preço de um computador é aditivo, a soma das peças, e essa é exatamente a forma que a Linear assume. Seu MAPE de ${N(L.MAPE)}% está praticamente no **piso teórico**: a variação de mercado de ±8% é aleatória e uniforme, com desvio absoluto médio de 4%, e nenhum modelo pode prevê-la a partir das peças.`),
  P(`**Não há sinais de sobreajuste.** O MAE da Linear é estável entre validação cruzada (${R(d.cv_mae.Linear)}), validação (${R(VL.MAE)}) e teste (${R(L.MAE)}), e o mesmo vale para o KNN (${R(d.cv_mae.KNN)}, ${R(VK.MAE)} e ${R(K.MAE)}). O desempenho inferior do KNN é, portanto, **viés** de representação, não variância: com One-Hot, sua distância conta quantas peças diferem, mas ignora quanto elas custam. Atributos numéricos de magnitude, como o nível de desempenho da peça, provavelmente reduziriam esse erro, o que fica como trabalho futuro.`),
  P("Quanto à **interpretabilidade**, a Linear é superior: explica globalmente o preço, com um valor em reais por peça. O KNN explica cada predição mostrando exemplos semelhantes, o que é intuitivo, mas não gera uma regra geral. Neste problema, portanto, o modelo mais simples é também o mais preciso e o mais explicável. A ressalva é a **multicolinearidade** da plataforma, que impede separar o custo de processador, placa-mãe e memória."),
  P("Em relação à literatura, o resultado difere de Tian (2024), em que a Linear obteve R² de apenas 0,62 com preços reais de notebooks. A explicação provável é a principal **limitação** deste trabalho: preços reais contêm efeitos não aditivos (marca, posicionamento, combinações), ausentes na simulação, que favorecem modelos não lineares. Validar os modelos com preços reais de montagens completas é o principal trabalho futuro."),
];

// ── 5 CONCLUSÃO ─────────────────────────────────────────────────────────
const s5 = [
  H1("5 CONCLUSÃO, PRODUTO FINAL E ENTREGÁVEIS"),
  H2("5.1 Conclusão"),
  P(`O objetivo geral foi atingido: os dois modelos foram desenvolvidos, comparados e explicados. A Regressão Linear Múltipla estimou o preço das montagens com MAE de ${R(L.MAE)} e R² de ${N(L.R2, 3)} no teste, próximo do limite imposto pelo ruído dos dados, e seus coeficientes recuperaram o preço das peças do catálogo. O KNN, otimizado por GridSearchCV (K = ${kp.K}), obteve MAE de ${R(K.MAE)}, limitado por uma distância que mede semelhança e não custo. Os resultados reforçam que a escolha do modelo deve considerar a estrutura do problema: quando a relação é aditiva, um modelo linear e interpretável é a melhor opção.`),
  P("Como trabalhos futuros, destacam-se: validar os modelos com preços reais de montagens completas; incluir atributos numéricos de desempenho para o KNN; atualizar periodicamente o catálogo, dada a volatilidade dos preços de hardware; e ampliar o catálogo de peças."),
  H2("5.2 Entregáveis obrigatórios"),
  AL("a)\t**Documento final:** este relatório, em formato ABNT."),
  AL("b)\t**Código-fonte:** repositório com os scripts que reproduzem todo o pipeline: `pc_gamer_ml.py` (catálogo, dados, divisão, EDA, treino e avaliação), `app_streamlit.py` e `tab_teoria.py` (protótipo de deployment) e `relatorio/gerar_figuras.py` e `relatorio/build.js` (figuras e geração deste relatório). Link: https://github.com/Loren1z9o/Price-Predictory-Hardware-."),
  AL("c)\t**Apresentação:** slides para a defesa oral do projeto."),
  AL("d)\t**Especificações técnicas:** Python 3.10 ou superior; bibliotecas streamlit (≥ 1.50), pandas (≥ 2.0), numpy (≥ 1.24), scikit-learn (≥ 1.3) , plotly (≥ 5.18) e matplotlib (≥ 3.7, só para as figuras do relatório), listadas em `requirements.txt`; qualquer computador com 4 GB de RAM. Instalação: `pip install -r requirements.txt`. Execução: `python pc_gamer_ml.py` (métricas no terminal) ou `python -m streamlit run app_streamlit.py` (interface). O treinamento completo leva cerca de 10 segundos, e os resultados são reprodutíveis pela semente fixa 42."),
  P("O protótipo em Streamlit tem três abas: **Preditor**, em que o usuário monta uma configuração compatível e recebe as estimativas dos dois modelos; **Resultados**, com as métricas, os gráficos e a EDA; e **Teoria**, que refaz com a configuração escolhida as demonstrações da Seção 4.3.3, provando numericamente como cada modelo calcula o preço."),
];

// ── REFERÊNCIAS ─────────────────────────────────────────────────────────
const refs = [
  "ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. **NBR 14724**: informação e documentação: trabalhos acadêmicos: apresentação. Rio de Janeiro: ABNT, 2011.",
  "ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. **NBR 6023**: informação e documentação: referências: elaboração. Rio de Janeiro: ABNT, 2018.",
  "BERNDT, E. R.; GRILICHES, Z.; RAPPAPORT, N. J. Econometric estimates of price indexes for personal computers in the 1990's. **Journal of Econometrics**, v. 68, p. 243-268, 1995.",
  "COVER, T.; HART, P. Nearest neighbor pattern classification. **IEEE Transactions on Information Theory**, v. 13, n. 1, p. 21-27, 1967.",
  "FOUTO, N. M. M. D.; DE ANGELO, C. F.; LUPPE, M. R. A five-year hedonic price breakdown for desktop personal computer attributes in Brazil. **BAR – Brazilian Administration Review**, Curitiba, v. 6, n. 3, p. 173-186, 2009. DOI: 10.1590/S1807-76922009000300002.",
  "HARDWARE BARATO. **Comparador de preços de hardware**. [S. l.], 2026. Disponível em: https://www.hardwarebarato.com. Acesso em: 27 ago. 2026.",
  "HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. **The elements of statistical learning**: data mining, inference, and prediction. 2. ed. New York: Springer, 2009.",
  "HILDEBRAND, Y. Jogador brasileiro está mais atento ao uso da IA em games, mostra pesquisa. **Tecnoblog**, 2026. Disponível em: https://tecnoblog.net/noticias/jogador-brasileiro-esta-mais-atento-ao-uso-da-ia-em-games-mostra-pesquisa/. Acesso em: 26 set. 2026.",
  "JAMES, G. et al. **An introduction to statistical learning**: with applications in R. New York: Springer, 2013.",
  "KAUFMAN, S. et al. Leakage in data mining: formulation, detection, and avoidance. **ACM Transactions on Knowledge Discovery from Data**, v. 6, n. 4, p. 1-21, 2012.",
  "KOHAVI, R. A study of cross-validation and bootstrap for accuracy estimation and model selection. In: INTERNATIONAL JOINT CONFERENCE ON ARTIFICIAL INTELLIGENCE, 14., 1995, Montreal. **Proceedings** [...]. San Francisco: Morgan Kaufmann, 1995. p. 1137-1143.",
  "MEIO & MENSAGEM. Indústria global de games atingirá US$ 188,8 bilhões em 2025. **Meio & Mensagem**, São Paulo, 15 set. 2025. Disponível em: https://www.meioemensagem.com.br/marketing/industria-global-de-games-atingira-us-1888-bi-em-2025. Acesso em: 26 set. 2026.",
  "PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825-2830, 2011.",
  "ROSEN, S. Hedonic prices and implicit markets: product differentiation in pure competition. **Journal of Political Economy**, v. 82, n. 1, p. 34-55, 1974.",
  "SIBURIAN, A. D. et al. Laptop price prediction with machine learning using regression algorithm. **Jurnal Sistem Informasi dan Ilmu Komputer Prima (JUSIKOM PRIMA)**, v. 6, n. 1, p. 87-91, 2022.",
  "TIAN, P. Research on laptop price predictive model based on linear regression, random forest and XGBoost. **Highlights in Science, Engineering and Technology**, v. 85, p. 265-271, 2024.",
];
const sRef = [
  new Paragraph({ pageBreakBefore: true, alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "REFERÊNCIAS", bold: true })] }),
  ...refs.map((r) => new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 240, line: 240 }, children: runs(r) })),
];

// ── documento ───────────────────────────────────────────────────────────
const page = { size: { width: 11906, height: 16838 },
  margin: { top: 1701, left: 1701, bottom: 1134, right: 1134, header: 709 } };
const hStyle = (id, name, lvl, size = 24) => ({ id, name, basedOn: "Normal", next: "Normal", quickFormat: true,
  run: { font: FONT, size, bold: true, allCaps: lvl === 0 },
  paragraph: { spacing: { before: 360, after: 240, line: 360 }, outlineLevel: lvl, keepNext: true } });

const doc = new Document({
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [hStyle("Heading1", "Heading 1", 0), hStyle("Heading2", "Heading 2", 1),
      { ...hStyle("Heading3", "Heading 3", 2), run: { font: FONT, size: 24, bold: false, italics: false } }],
  },
  sections: [
    { properties: { page }, children: [...capa, ...rosto, ...resumo, ...sumario] },
    { properties: { page },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
      children: [...s1, ...s2, ...s3, ...s4, ...s5, ...sRef] },
  ],
});
Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync("Relatorio_PC_Gamer_Predictor_v7.docx", b);
  console.log(`ok · ${nFig} figuras · ${nTab} tabelas`);
});
