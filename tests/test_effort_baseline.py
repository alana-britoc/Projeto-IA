import numpy as np
import pandas as pd
import pytest

from src.data import loader
from src.evaluation import effort_baseline
from src.evaluation.metrics import popt


def test_referencia_e_revisar_do_menor_para_o_maior():
    # sem modelo, a densidade de risco so depende do tamanho: menor commit primeiro
    y = np.array([0, 1, 0, 1])
    effort = np.array([50, 1, 10, 2])
    ref = effort_baseline.size_only(y, effort)
    ordem_por_tamanho = popt(y, 1 / np.maximum(effort, 1), effort)
    assert ref["popt"] == pytest.approx(ordem_por_tamanho)
    # aqui os dois commits com defeito sao os menores, entao a referencia ja e a ordem otima
    assert ref["popt"] == pytest.approx(1.0)


@pytest.fixture
def raw(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for p in ("qt", "go"):
        (tmp_path / p / "cross").mkdir(parents=True)
        n = 50
        df = pd.DataFrame({f: 1.0 for f in loader.FEATURES}, index=range(n))
        df["_id"] = [f"{p}{i}" for i in range(n)]
        df["date"] = range(n)
        df["bug"] = rng.integers(0, 2, size=n)
        df["la"] = rng.integers(0, 100, size=n)
        df["ld"] = rng.integers(0, 20, size=n)
        df.to_csv(tmp_path / p / f"{p}_k_feature.csv", index=False)
        df.iloc[:40].to_csv(tmp_path / p / "cross" / "k_train.csv", index=False)
        df.iloc[40:].to_csv(tmp_path / p / "cross" / "k_test.csv", index=False)
    monkeypatch.setattr(loader, "RAW", tmp_path)
    return tmp_path


def test_referencia_cobre_o_cp_crono():
    # #24: o CP_crono tem o mesmo teste do CP, mas precisa aparecer na tabela
    assert effort_baseline.SETTINGS == loader.SETTINGS


def test_tabela_tem_um_valor_por_projeto_e_cenario(raw):
    df = effort_baseline.reference_table(["qt", "go"], ["WP", "CP"])
    assert list(df.columns) == ["project", "setting", "popt", "recall_20"]
    assert len(df) == 4
    assert df[["popt", "recall_20"]].apply(lambda c: c.between(0, 1)).all().all()
