import pytest

from src.data.loader import ROOT
from src.models import simulator


def _sklearn_fixado(path):
    for linha in path.read_text(encoding="utf-8").splitlines():
        if linha.startswith("scikit-learn=="):
            return linha.split("==")[1].strip()
    return None


@pytest.mark.parametrize("arquivo", ["requirements.txt", "src/requirements.txt"])
def test_sklearn_fixado_igual_ao_do_modelo_salvo(arquivo):
    # se treinar o simulador com outra versao, tem que atualizar aqui tambem
    assert _sklearn_fixado(ROOT / arquivo) == simulator.load()["sklearn_version"]


def test_deploy_tem_o_que_o_painel_usa():
    libs = (ROOT / "src" / "requirements.txt").read_text(encoding="utf-8").split()
    nomes = {lib.split("==")[0] for lib in libs}
    assert {"streamlit", "scikit-learn", "pandas", "matplotlib"} <= nomes
    assert "jupyter" not in nomes
