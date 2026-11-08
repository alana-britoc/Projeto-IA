import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import dashboard
from src.data.loader import PROJECTS, ROOT
from src.evaluation.plots import REFERENCE_MARKERS, model_colors


@pytest.fixture(scope="module")
def data():
    return dashboard.load_comparison()


def test_carrega_todos_os_modelos_versionados(data):
    comp, _, order = data
    assert order[0] == "lapredict"
    assert set(comp["model"]) == set(order)
    assert {"random_forest", "svm", "deepjit", "dbn_jit"} <= set(order)


def test_resultados_seguem_a_ordem_dos_modelos(data):
    comp, _, order = data
    assert list(dict.fromkeys(comp["model"])) == order
    first = comp.head(12)
    assert set(first["model"]) == {"lapredict"}
    assert list(dict.fromkeys(first["project"])) == PROJECTS


def test_testes_batem_com_o_csv_da_comparacao(data):
    # o painel recalcula com as funcoes da #7, entao tem que dar o mesmo que foi versionado
    _, tests, _ = data
    saved = pd.read_csv(ROOT / "results" / "metrics" / "comparacao_testes.csv")
    keys = ["metric", "model"]
    merged = tests.merge(saved, on=keys, suffixes=("", "_csv"))
    assert len(merged) == len(saved) == len(tests)
    assert (merged["verdict"] == merged["verdict_csv"]).all()
    assert merged["p_holm"].to_numpy() == pytest.approx(merged["p_holm_csv"].to_numpy())


def test_resposta_igual_ao_relatorio(data):
    _, tests, _ = data
    report = (ROOT / "results" / "comparacao.md").read_text(encoding="utf-8")
    assert dashboard.headline(tests) in report
    assert dashboard.headline(tests).startswith(("**Sim", "**Não"))


def test_filtro_por_modelo_projeto_e_cenario(data):
    comp, _, _ = data
    out = dashboard.filter_comparison(comp, ["lapredict", "svm"], ["qt", "go"], ["WP"])
    assert set(out["model"]) == {"lapredict", "svm"}
    assert set(out["project"]) == {"qt", "go"}
    assert set(out["setting"]) == {"WP"}
    assert len(out) == 4


def test_media_por_modelo_ignora_quem_nao_tem_a_metrica(data):
    comp, _, _ = data
    out = dashboard.mean_by_model(comp, "f1")
    assert "dbn_jit" not in set(out["model"])
    lap = out[(out["model"] == "lapredict") & (out["setting"] == "WP")]["valor"]
    assert lap.item() == pytest.approx(0.4122, abs=1e-4)


def test_media_da_auc_do_lapredict_igual_ao_artigo(data):
    comp, _, _ = data
    out = dashboard.mean_by_model(comp, "auc")
    assert out[out["model"] == "lapredict"]["valor"].to_numpy() == pytest.approx(0.7246, abs=1e-4)


def test_tabela_de_testes_so_da_metrica_e_dos_modelos_escolhidos(data):
    _, tests, _ = data
    out = dashboard.tests_table(tests, "auc", ["random_forest", "svm"])
    assert list(out["Modelo"]) == ["random_forest", "svm"]
    assert "Resultado" in out.columns


def test_graficos_usam_as_cores_das_figuras(data):
    comp, _, order = data
    spec = dashboard.project_chart(comp, "auc", order).to_dict()
    scale = spec["encoding"]["color"]["scale"]
    colors = model_colors(order)
    assert dict(zip(scale["domain"], scale["range"])) == colors
    assert dashboard.mean_chart(comp, "auc", order).to_dict()


def test_modelos_do_artigo_tem_forma_propria_no_grafico(data):
    # os modelos do artigo sao cinza, entao a forma e que separa um do outro
    comp, _, order = data
    shape = dashboard.project_chart(comp, "auc", order).to_dict()["encoding"]["shape"]["scale"]
    shapes = dict(zip(shape["domain"], shape["range"]))
    assert {shapes[m] for m in REFERENCE_MARKERS} == set(REFERENCE_MARKERS.values())
    assert {shapes[m] for m in order if m not in REFERENCE_MARKERS} == {"circle"}


def test_app_abre_sem_erro_e_troca_a_metrica():
    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.sidebar.selectbox[0].set_value("mcc").run()
    assert not at.exception
    assert any("MCC" in h.value for h in at.subheader)
