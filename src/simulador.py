"""Pagina do simulador de commit (#20). Abre pelo src/app.py."""
import altair as alt
import sklearn
import streamlit as st

from src.data.loader import PROJECTS
from src.evaluation.plots import PALETTE
from src.models import simulator


@st.cache_resource
def carregar():
    bundle = simulator.load()
    return bundle, simulator.load_explainer(bundle)


bundle, explicador = carregar()

st.title("Simulador de commit")
st.warning("**Demonstração.** O modelo foi treinado com os commits de 6 projetos de código "
           f"aberto ({', '.join(PROJECTS)}). A probabilidade não tem validade para commits de "
           "outros projetos.")
if not simulator.same_sklearn(bundle):
    st.warning(f"O modelo foi salvo com o scikit-learn {bundle['sklearn_version']} e você está "
               f"usando o {sklearn.__version__}. Se der erro, treine de novo com "
               "`python -m src.models.train_simulator`.")

st.markdown("Preencha as métricas do commit. Os valores iniciais são a mediana dos commits "
            "usados no treino.")


def campo(feature):
    label, inteiro, minimo, ajuda = simulator.FIELDS[feature]
    mediana = bundle["medians"][feature]
    key = f"sim_{feature}"
    if feature == "fix":
        return st.checkbox(label, value=mediana >= 0.5, help=ajuda, key=key)
    if inteiro:
        return st.number_input(label, min_value=minimo, value=int(round(mediana)), step=1,
                               help=ajuda, key=key)
    return st.number_input(label, min_value=minimo, value=round(mediana, 2), step=0.1,
                           format="%.2f", help=ajuda, key=key)


form, resultado = st.columns([3, 2], gap="large")
values = {}
with form:
    esquerda, direita = st.columns(2)
    grupos = list(simulator.GROUPS.items())
    for i, (grupo, features) in enumerate(grupos):
        with esquerda if i < 2 else direita:
            st.subheader(grupo)
            for feature in features:
                values[feature] = campo(feature)

prob = simulator.predict(bundle, values)
pct = simulator.percentile(bundle, prob)

with resultado:
    st.subheader("Resultado")
    st.metric("Probabilidade de introduzir defeito", f"{prob:.0%}")
    st.progress(prob)
    st.metric("Mais arriscado que", f"{pct:.0f}% dos commits")
    st.caption("A probabilidade é calibrada: entre os commits que o modelo marca com 30%, "
               "cerca de 30% tinham defeito de verdade. Nos commits mais recentes dos 6 "
               f"projetos, {bundle['defect_rate']:.0%} tinham defeito.")

    with st.expander("Sobre o modelo"):
        params = bundle["params"]
        n = {k: f"{bundle[k]:,}".replace(",", ".") for k in ("n_train", "n_calib", "n_eval")}
        brier = bundle["brier"]
        st.markdown(
            f"- Random Forest com {params['n_estimators']} árvores de profundidade até "
            f"{params['max_depth']}, calibrado com o método {bundle['method']}\n"
            f"- {n['n_train']} commits de treino, {n['n_calib']} de calibração e "
            f"{n['n_eval']} de avaliação (80%, 10% e 10% de cada projeto, em ordem de data)\n"
            f"- Brier score na avaliação: {brier['sem calibração']:.3f} sem calibração e "
            f"{brier[bundle['method']]:.3f} calibrado (menor é melhor)\n"
            f"- AUC na avaliação: {bundle['auc']:.3f}\n"
            f"- scikit-learn {bundle['sklearn_version']}\n\n"
            "Os outros hiperparâmetros são os que a busca escolheu mais vezes nos "
            "experimentos. Para treinar de novo: `python -m src.models.train_simulator`.")


def rotulo(linha):
    if linha.feature == "fix":
        return f"{linha.nome}: {'sim' if linha.valor else 'não'}"
    inteiro = simulator.FIELDS[linha.feature][1]
    valor = f"{linha.valor:.0f}" if inteiro else f"{linha.valor:.2f}"
    return f"{linha.nome}: {valor}"


contrib, base = simulator.contributions(explicador, bundle, values)
contrib["pontos"] = 100 * contrib["contribuicao"]
contrib["rotulo"] = [rotulo(linha) for linha in contrib.itertuples()]
contrib["efeito"] = ["aumenta o risco" if p > 0 else "diminui o risco" for p in contrib["pontos"]]
score = base + contrib["contribuicao"].sum()

st.subheader("O que mais pesou neste commit")
grafico = alt.Chart(contrib).mark_bar().encode(
    x=alt.X("pontos:Q", title="pontos no score do modelo"),
    y=alt.Y("rotulo:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
    color=alt.Color("efeito:N", title=None,
                    scale=alt.Scale(domain=["aumenta o risco", "diminui o risco"],
                                    range=[PALETTE[1], PALETTE[0]]),
                    legend=alt.Legend(orient="bottom")),
    tooltip=[alt.Tooltip("nome:N", title="Métrica"), alt.Tooltip("valor:Q", title="Valor"),
             alt.Tooltip("pontos:Q", title="Pontos", format="+.1f")],
)
st.altair_chart(grafico, width="stretch")
st.caption(f"Contribuição de cada métrica calculada com SHAP. O score do Random Forest parte "
           f"de {100 * base:.0f} pontos, o mesmo para qualquer commit, e as barras somadas levam "
           f"até {100 * score:.0f} pontos. A calibração transforma esse score na probabilidade "
           "acima sem mudar a ordem: o que aumenta o score também aumenta a probabilidade.")
