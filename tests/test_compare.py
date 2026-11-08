import pandas as pd
import pytest

from src.data.loader import PROJECTS
from src.evaluation.compare import (ARTIGO, METRICS, REALISTA, aggregate, all_tests, answer,
                                    available_comparisons, build_comparison,
                                    compare_to_baseline, deep_results, load_results,
                                    metric_table, paper_results, save_outputs, summary_table,
                                    verdict)


def _rows(model, auc_by_seed, project="qt", setting="WP", esforco=True):
    rows = [{"model": model, "project": project, "setting": setting, "seed": s,
             "auc": a, "pr_auc": a / 2, "f1": a / 3, "mcc": a / 4}
            for s, a in enumerate(auc_by_seed)]
    if esforco:
        for r in rows:
            r.update(popt=r["auc"] * 0.9, recall_20=r["auc"] / 5)
    return rows


def test_metricas_incluem_as_de_esforco():
    assert METRICS == ["auc", "pr_auc", "f1", "mcc", "popt", "recall_20"]


def test_csv_antigo_sem_esforco_nao_quebra_a_comparacao():
    # resultados rodados antes da #25 nao tem popt nem recall_20
    rows = []
    for p in PROJECTS:
        for s in ("WP", "CP"):
            rows += (_rows("lapredict", [0.7], p, s, esforco=False)
                     + _rows("bom", [0.8], p, s, esforco=False))
    comp = build_comparison(pd.DataFrame(rows))
    assert comp["popt"].isna().all()
    tests = all_tests(comp)
    assert "popt" not in set(tests["metric"])
    assert "auc" in set(tests["metric"])


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


def _deep_csv(tmp_path):
    path = tmp_path / "deep_baselines.csv"
    pd.DataFrame([{"model": "deepjit", "project": "qt", "setting": "WP", "auc": 0.69,
                   "pr_auc": 0.27}]).to_csv(path, index=False)
    return path


def test_deep_results_tem_auc_e_pr_auc(tmp_path):
    df = deep_results(_deep_csv(tmp_path))
    assert df.loc[0, "pr_auc"] == pytest.approx(0.27)
    assert df[["f1", "mcc"]].isna().all().all()
    assert (df["source"] == "autores").all()


def test_deep_results_sem_arquivo_fica_vazio(tmp_path):
    assert deep_results(tmp_path / "nao_existe.csv").empty


def test_build_comparison_prefere_as_predicoes_dos_autores_ao_artigo(tmp_path):
    comp = build_comparison(pd.DataFrame(_rows("lapredict", [0.7])), deep_path=_deep_csv(tmp_path))
    deepjit = comp[comp["model"] == "deepjit"]
    assert len(deepjit) == 1
    assert deepjit["source"].iloc[0] == "autores"
    # cc2vec nao ta no csv, entao vem do artigo
    assert set(comp.loc[comp["model"] == "cc2vec", "source"]) == {"artigo"}


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


def test_summary_table_tem_todas_as_metricas():
    s = summary_table(_comp())
    assert {m for m, _ in s.columns} == set(METRICS)
    assert s.loc["bom", ("auc", "CP")] == pytest.approx(0.80)


def _comp_com_ruido():
    # "bom" ganha do lapredict em todos os 12 pares, "igual" empata
    rows = []
    for i, p in enumerate(PROJECTS):
        for j, s in enumerate(("WP", "CP")):
            base = 0.65 + 0.01 * i + 0.003 * j
            rows += (_rows("lapredict", [base], p, s) + _rows("bom", [base + 0.05 + 0.001 * i], p, s)
                     + _rows("igual", [base], p, s))
    return build_comparison(pd.DataFrame(rows))


def test_compare_to_baseline_detecta_quem_supera():
    t = compare_to_baseline(_comp_com_ruido(), "auc").set_index("model")
    assert t.loc["bom", "n"] == 12
    assert t.loc["bom", "p_holm"] < 0.05
    assert t.loc["bom", "delta"] > 0
    assert t.loc["igual", "p_value"] == 1.0
    # modelos do artigo entram na AUC
    assert {"deepjit", "cc2vec", "dbn_jit"} <= set(t.index)
    assert "lapredict" not in t.index


def test_compare_to_baseline_pula_modelo_sem_a_metrica():
    t = compare_to_baseline(_comp_com_ruido(), "mcc")
    assert set(t["model"]) == {"bom", "igual"}


def test_compare_to_baseline_sem_baseline_quebra():
    comp = build_comparison(pd.DataFrame(_rows("bom", [0.8])))
    with pytest.raises(KeyError):
        compare_to_baseline(comp, "auc")


@pytest.mark.parametrize("p_holm, delta, magnitude, esperado", [
    (0.01, 0.5, "grande", "supera"),
    (0.01, -0.5, "grande", "perde"),
    (0.20, 0.5, "grande", "empate"),
    (0.01, 0.1, "negligível", "empate"),
    (float("nan"), 0.5, "grande", "empate"),
])
def test_verdict(p_holm, delta, magnitude, esperado):
    assert verdict({"p_holm": p_holm, "delta": delta, "magnitude": magnitude}) == esperado


def test_all_tests_cobre_todas_as_metricas():
    t = all_tests(_comp_com_ruido())
    assert set(t["metric"]) == set(METRICS)
    assert t.set_index(["metric", "model"]).loc[("f1", "bom"), "verdict"] == "supera"


def test_answer_cita_quem_supera():
    text = answer(all_tests(_comp_com_ruido()))
    assert "**Sim:**" in text
    assert "bom em AUC" in text
    assert "| bom | MCC |" in text


def test_answer_sem_vencedor_diz_nao():
    rows = []
    for p in PROJECTS:
        for s in ("WP", "CP"):
            rows += _rows("lapredict", [0.7], p, s) + _rows("igual", [0.7], p, s)
    text = answer(all_tests(build_comparison(pd.DataFrame(rows))))
    assert text.startswith("## Algum modelo supera o lapredict?")
    assert "**Não.**" in text


def test_save_outputs_gera_tabelas_e_relatorio(tmp_path):
    comp = _comp_com_ruido()
    paths = save_outputs(comp, all_tests(comp), tmp_path, tmp_path / "comparacao.md")
    names = {p.name for p in paths}
    assert {"comparacao_resultados.csv", "comparacao_resumo.csv", "comparacao_testes.csv",
            "comparacao_auc.csv", "comparacao_pr_auc.csv", "comparacao_f1.csv",
            "comparacao_mcc.csv", "comparacao_popt.csv", "comparacao_recall_20.csv",
            "comparacao.md"} == names
    # o load_results nao pode confundir as tabelas da comparacao com experimentos
    assert not list(tmp_path.glob("experimentos_*.csv"))
    report = (tmp_path / "comparacao.md").read_text(encoding="utf-8")
    assert "## Algum modelo supera o lapredict?" in report
    assert "| bom |" in report
    assert "só a AUC publicada" in report


def test_relatorio_mostra_a_referencia_de_esforco_quando_existe(tmp_path):
    # #25: sem a referencia, Popt alto parece merito do modelo
    pd.DataFrame({"project": ["qt", "qt"], "setting": ["WP", "CP"], "popt": [0.8, 0.8],
                  "recall_20": [0.6, 0.6]}).to_csv(tmp_path / "esforco_referencia.csv",
                                                   index=False)
    comp = _comp_com_ruido()
    save_outputs(comp, all_tests(comp), tmp_path, tmp_path / "comparacao.md")
    report = (tmp_path / "comparacao.md").read_text(encoding="utf-8")
    assert "Referência sem modelo" in report
    assert "0.800" in report and "0.600" in report


def test_relatorio_sem_referencia_de_esforco_nao_quebra(tmp_path):
    comp = _comp_com_ruido()
    save_outputs(comp, all_tests(comp), tmp_path, tmp_path / "comparacao.md")
    assert "Referência sem modelo" not in (tmp_path / "comparacao.md").read_text(encoding="utf-8")


# cenario realista (#24): WP + CP_crono no lugar de WP + CP
def _comp_com_crono():
    # "bom" ganha nos 3 cenarios; "fragil" ganha no WP e no CP, mas perde no CP_crono
    rows = []
    for i, p in enumerate(PROJECTS):
        for j, s in enumerate(("WP", "CP", "CP_crono")):
            base = 0.65 + 0.01 * i + 0.003 * j
            fragil = base - 0.04 - 0.001 * i if s == "CP_crono" else base + 0.04 + 0.001 * i
            rows += (_rows("lapredict", [base], p, s) + _rows("bom", [base + 0.05 + 0.001 * i], p, s)
                     + _rows("fragil", [fragil], p, s))
    return build_comparison(pd.DataFrame(rows))


def test_cada_comparacao_usa_so_os_seus_cenarios():
    comp = _comp_com_crono()
    for settings in (ARTIGO, REALISTA):
        t = compare_to_baseline(comp, "auc", settings=settings).set_index("model")
        assert t.loc["bom", "n"] == 12


def test_modelo_que_so_ganha_com_treino_do_futuro_nao_supera_no_realista():
    comp = _comp_com_crono()
    artigo = all_tests(comp).set_index(["metric", "model"])
    realista = all_tests(comp, settings=REALISTA).set_index(["metric", "model"])
    assert artigo.loc[("auc", "fragil"), "verdict"] == "supera"
    assert realista.loc[("auc", "fragil"), "verdict"] != "supera"
    assert realista.loc[("auc", "bom"), "verdict"] == "supera"
    assert set(artigo["cenarios"]) == {"WP+CP"}
    assert set(realista["cenarios"]) == {"WP+CP_crono"}


def test_comparacao_realista_so_existe_com_cp_crono():
    assert available_comparisons(_comp_com_ruido()) == [ARTIGO]
    assert available_comparisons(_comp_com_crono()) == [ARTIGO, REALISTA]


def test_tabelas_incluem_o_cp_crono_depois_do_cp():
    comp = _comp_com_crono()
    s = summary_table(comp)
    assert [c for c in s.columns if c[0] == "auc"] == [("auc", "WP"), ("auc", "CP"),
                                                        ("auc", "CP_crono")]
    t = metric_table(comp, "auc")
    assert ("CP_crono", "media") in t.columns
    assert list(dict.fromkeys(t.columns.get_level_values(0))) == ["WP", "CP", "CP_crono"]


def test_resposta_do_cenario_realista_tem_titulo_proprio():
    comp = _comp_com_crono()
    texto = answer(all_tests(comp, settings=REALISTA))
    assert texto.startswith("## Algum modelo supera o lapredict no cenário realista")
    assert texto.splitlines()[2].startswith("**Sim:**")


def test_save_outputs_inclui_as_duas_comparacoes(tmp_path):
    comp = _comp_com_crono()
    tests = pd.concat([all_tests(comp, settings=s) for s in available_comparisons(comp)])
    save_outputs(comp, tests, tmp_path, tmp_path / "comparacao.md")
    report = (tmp_path / "comparacao.md").read_text(encoding="utf-8")
    assert "## Algum modelo supera o lapredict?" in report
    assert "## Algum modelo supera o lapredict no cenário realista" in report
    saved = pd.read_csv(tmp_path / "comparacao_testes.csv")
    assert set(saved["cenarios"]) == {"WP+CP", "WP+CP_crono"}
