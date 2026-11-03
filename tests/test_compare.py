import pandas as pd
import pytest

from src.data.loader import PROJECTS
from src.evaluation.compare import (aggregate, build_comparison, load_results, metric_table,
                                    paper_results, summary_table)


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


def test_paper_results_so_tem_auc():
    df = paper_results()
    assert set(df["model"]) == {"deepjit", "cc2vec", "dbn_jit"}
    # 6 projetos x 2 cenarios, sem a linha de media
    assert (df.groupby("model").size() == 12).all()
    assert "Mean" not in set(df["project"])
    assert df["auc"].notna().all()
    assert df[["pr_auc", "f1", "mcc"]].isna().all().all()
    assert (df["source"] == "artigo").all()


def test_build_comparison_prefere_o_modelo_rodado_por_nos():
    ours = pd.DataFrame(_rows("deepjit", [0.9]) + _rows("lapredict", [0.7]))
    comp = build_comparison(ours)
    deepjit = comp[comp["model"] == "deepjit"]
    assert len(deepjit) == 1
    assert deepjit["source"].iloc[0] == "nosso"
    assert set(comp["model"]) == {"deepjit", "lapredict", "cc2vec", "dbn_jit"}


def _comp():
    rows = []
    for p in PROJECTS:
        for s in ("WP", "CP"):
            rows += _rows("lapredict", [0.70], p, s) + _rows("bom", [0.80], p, s)
    return build_comparison(pd.DataFrame(rows))


def test_metric_table_tem_media_por_cenario_e_baseline_primeiro():
    t = metric_table(_comp(), "auc")
    assert t.index[0] == "lapredict"
    assert ("WP", "media") in t.columns and ("CP", "media") in t.columns
    assert t.loc["bom", ("WP", "media")] == pytest.approx(0.80)
    # colunas na ordem: WP antes de CP, media por ultimo
    assert list(t.columns) == ([("WP", p) for p in PROJECTS] + [("WP", "media")]
                               + [("CP", p) for p in PROJECTS] + [("CP", "media")])


def test_metric_table_descarta_modelo_sem_a_metrica():
    # os modelos do artigo nao tem mcc
    t = metric_table(_comp(), "mcc")
    assert set(t.index) == {"lapredict", "bom"}


def test_summary_table_tem_as_quatro_metricas():
    s = summary_table(_comp())
    assert {m for m, _ in s.columns} == {"auc", "pr_auc", "f1", "mcc"}
    assert s.loc["bom", ("auc", "CP")] == pytest.approx(0.80)
