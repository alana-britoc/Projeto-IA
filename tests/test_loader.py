import numpy as np
import pandas as pd
import pytest

from src.data import loader


def _commits(n, start, la):
    # CSV no formato do dataset: _id, date, bug e as 14 features
    df = pd.DataFrame({f: 1.0 for f in loader.FEATURES}, index=range(n))
    df["_id"] = [f"c{start + i}" for i in range(n)]
    df["date"] = [start + i for i in range(n)]
    df["bug"] = [i % 2 for i in range(n)]
    df["la"] = la
    df["ld"] = 2.0
    return df


@pytest.fixture
def raw(tmp_path, monkeypatch):
    (tmp_path / "qt" / "cross").mkdir(parents=True)
    _commits(10, 0, np.arange(10.0)).to_csv(tmp_path / "qt" / "qt_k_feature.csv", index=False)
    _commits(6, 100, 5.0).to_csv(tmp_path / "qt" / "cross" / "k_train.csv", index=False)
    _commits(4, 200, 7.0).to_csv(tmp_path / "qt" / "cross" / "k_test.csv", index=False)
    monkeypatch.setattr(loader, "RAW", tmp_path)
    return tmp_path


def test_wp_separa_80_20_em_ordem(raw):
    d = loader.load_split("qt", "WP")
    assert len(d["y_train"]) == 8 and len(d["y_test"]) == 2
    assert list(d["ids_test"]) == ["c8", "c9"]


def test_esforco_do_teste_e_la_mais_ld(raw):
    # #25: o esforco vem das linhas do commit, mesmo quando o modelo so usa o la (LApredict)
    d = loader.load_split("qt", "WP", features=["la"])
    assert list(d["effort_test"]) == [8 + 2, 9 + 2]
    assert list(loader.load_split("qt", "CP")["effort_test"]) == [9.0] * 4


def test_setting_invalido_da_erro(raw):
    with pytest.raises(ValueError):
        loader.load_split("qt", "XX")
