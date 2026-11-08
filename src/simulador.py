"""Pagina do simulador de commit (#20). Abre pelo src/app.py."""
import sklearn
import streamlit as st

from src.data.loader import PROJECTS
from src.models import simulator


@st.cache_resource
def carregar():
    return simulator.load()


bundle = carregar()

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
