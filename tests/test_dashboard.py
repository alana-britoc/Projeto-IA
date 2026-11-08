import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import dashboard
from src.data.loader import PROJECTS, ROOT, SETTINGS
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
    # 6 projetos x cenarios que existem (3 com o CP_crono da #24)
    first = comp.head(len(PROJECTS) * comp["setting"].nunique())
    assert set(first["model"]) == {"lapredict"}
    assert list(dict.fromkeys(first["project"])) == PROJECTS
    assert list(dict.fromkeys(first["setting"])) == SETTINGS


def test_testes_batem_com_o_csv_da_comparacao(data):
    # o painel recalcula com as funcoes da #7, entao tem que dar o mesmo que foi versionado
    _, tests, _ = data
    saved = pd.read_csv(ROOT / "results" / "metrics" / "comparacao_testes.csv")
    keys = ["cenarios", "metric", "model"]
    merged = tests.merge(saved, on=keys, suffixes=("", "_csv"))
    assert len(merged) == len(saved) == len(tests)
    assert (merged["verdict"] == merged["verdict_csv"]).all()
    assert merged["p_holm"].to_numpy() == pytest.approx(merged["p_holm_csv"].to_numpy())


def test_resposta_igual_ao_relatorio(data):
    _, tests, _ = data
    report = (ROOT / "results" / "comparacao.md").read_text(encoding="utf-8")
    for _, group in tests.groupby("cenarios"):
        assert dashboard.headline(group) in report
        assert dashboard.headline(group).startswith(("**Sim", "**Não"))


def test_painel_tem_o_cp_crono_e_as_duas_comparacoes(data):
    # #24: o cenario realista entra nos filtros e na comparacao estatistica
    comp, tests, _ = data
    assert dashboard.SETTINGS == SETTINGS
    assert set(dashboard.SETTING_NAMES) == set(SETTINGS)
    assert "CP_crono" in set(comp["setting"])
    assert list(dict.fromkeys(tests["cenarios"])) == list(dashboard.COMPARISON_NAMES)


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
    # como na pagina: a tabela usa so os testes da comparacao escolhida
    tests = tests[tests["cenarios"] == "WP+CP"]
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


def test_app_troca_para_a_comparacao_realista():
    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("WP+CP_crono").run()
    assert not at.exception
    assert any("cenário realista" in h.value for h in at.subheader)
