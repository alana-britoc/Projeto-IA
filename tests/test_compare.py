import pandas as pd
import pytest

from src.evaluation.compare import aggregate, load_results


def _rows(model, auc_by_seed, project="qt", setting="WP"):
    return [{"model": model, "project": project, "setting": setting, "seed": s,
             "auc": a, "pr_auc": a / 2, "f1": a / 3, "mcc": a / 4}
            for s, a in enumerate(auc_by_seed)]


def test_load_results_le_todos_os_csvs(tmp_path):
    pd.DataFrame(_rows("a", [0.7])).to_csv(tmp_path / "experimentos_a.csv", index=False)
    pd.DataFrame(_rows("b", [0.6])).to_csv(tmp_path / "experimentos_b.csv", index=False)
    pd.DataFrame(_rows("c", [0.5])).to_csv(tmp_path / "reproducao.csv", index=False)
    df = load_results(tmp_path)
    assert sorted(df["model"].unique()) == ["a", "b"]


def test_load_results_sem_arquivos_quebra(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path)


def test_aggregate_tira_a_media_das_seeds():
    df = pd.DataFrame(_rows("a", [0.6, 0.8]) + _rows("a", [0.5], setting="CP"))
    agg = aggregate(df).set_index("setting")
    assert len(agg) == 2
    assert agg.loc["WP", "auc"] == pytest.approx(0.7)
    assert agg.loc["WP", "mcc"] == pytest.approx(0.175)
    assert (agg["source"] == "nosso").all()
