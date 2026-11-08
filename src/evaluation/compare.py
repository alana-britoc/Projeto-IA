"""Comparacao final (#7): junta os resultados de todos os modelos e compara com o LApredict.

Le todos os results/metrics/experimentos_*.csv, entao modelo novo entra sozinho.

Uso: python -m src.evaluation.compare
Saida: results/metrics/comparacao_*.csv, results/figures/*.png e results/comparacao.md
"""
import argparse

import pandas as pd

from src.data.loader import PROJECTS, ROOT, SETTINGS
from src.evaluation import plots
from src.evaluation.metrics import METRIC_NAMES
from src.evaluation.reproduce import REFERENCE, load_reference
from src.evaluation.stats import holm, paired_test

METRICS_DIR = ROOT / "results" / "metrics"
DEEP = METRICS_DIR / "deep_baselines.csv"
# todas as metricas de metrics.py, incluindo as de esforco da #25 (popt e recall_20)
METRICS = list(METRIC_NAMES)
KEYS = ["model", "project", "setting"]
BASELINE = "lapredict"
ALPHA = 0.05
# cenarios de cada comparacao (#24). A do artigo usa o CP com treino do futuro; a realista troca
# pelo CP_crono. O CP_crono sozinho daria so 6 pares, e com 6 pares o menor p do Wilcoxon e
# 0,031, que nao passa no Holm com varios modelos. Com o WP junto, sao 12 pares nas duas
ARTIGO = ("WP", "CP")
REALISTA = ("WP", "CP_crono")
# modelos do artigo que a gente nao roda. LApredict e LR-JIT ficam de fora pq ja temos os nossos
PAPER_MODELS = {"DeepJIT": "deepjit", "CC2Vec": "cc2vec", "DBN-JIT": "dbn_jit"}


def load_results(metrics_dir=METRICS_DIR):
    files = sorted(metrics_dir.glob("experimentos_*.csv"))
    if not files:
        raise FileNotFoundError(f"nenhum experimentos_*.csv em {metrics_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def aggregate(results):
    # media das seeds: o par do teste estatistico e projeto x cenario
    # (o lapredict nao tem busca, entao as seeds dele sao identicas)
    # CSV rodado antes de uma metrica existir fica com ela vazia (NaN), em vez de quebrar
    results = results.reindex(columns=list(dict.fromkeys([*results.columns, *METRICS])))
    agg = results.groupby(KEYS, as_index=False)[METRICS].mean()
    agg["source"] = "nosso"
    return agg


def deep_results(path=DEEP):
    # so AUC e PR-AUC, sem validacao nao tem como escolher o limiar
    if not path.exists():
        return pd.DataFrame(columns=KEYS + METRICS + ["source"])
    df = pd.read_csv(path).reindex(columns=KEYS + METRICS)
    df["source"] = "autores"
    return df


def paper_results(skip=(), path=REFERENCE):
    # o artigo so publica AUC
    ref = load_reference(path)
    rows = []
    for paper_name, model in PAPER_MODELS.items():
        if model in skip:
            continue
        for col, auc in ref[paper_name].items():
            project, setting = col.rsplit("_", 1)
            if project == "Mean":
                continue
            rows.append({"model": model, "project": project, "setting": setting, "auc": auc})
    df = pd.DataFrame(rows, columns=KEYS + ["auc"]).reindex(columns=KEYS + METRICS)
    df["source"] = "artigo"
    return df


def build_comparison(results, path=REFERENCE, deep_path=DEEP):
    ours = aggregate(results)
    # nosso > predicoes dos autores > tabela do artigo
    deep = deep_results(deep_path)
    deep = deep[~deep["model"].isin(ours["model"])]
    paper = paper_results(skip=set(ours["model"]) | set(deep["model"]), path=path)
    parts = [df for df in (ours, deep, paper) if not df.empty]
    return pd.concat(parts, ignore_index=True)


def model_order(comp, baseline="lapredict"):
    # baseline primeiro, depois por AUC media decrescente
    mean_auc = comp.groupby("model")["auc"].mean().sort_values(ascending=False)
    others = [m for m in mean_auc.index if m != baseline]
    return ([baseline] if baseline in mean_auc.index else []) + others


def metric_table(comp, metric):
    """Uma linha por modelo, uma coluna por projeto x cenario, mais a media de cada cenario."""
    table = comp.pivot_table(index="model", columns=["setting", "project"], values=metric,
                             dropna=False)
    table = table.dropna(how="all")
    for setting in SETTINGS:
        if setting in table.columns.get_level_values(0):
            table[(setting, "media")] = table[setting].mean(axis=1, skipna=False)
    table = table.reindex(columns=sorted(table.columns, key=_column_key))
    return table.reindex([m for m in model_order(comp) if m in table.index])


def _column_key(col):
    setting, project = col
    order = PROJECTS + ["media"]
    return (SETTINGS.index(setting), order.index(project) if project in order else 99)


def summary_table(comp):
    """Media das 6 bases por modelo e cenario, com as metricas lado a lado."""
    summ = comp.groupby(["model", "setting"])[METRICS].mean().unstack("setting")
    present = [s for s in SETTINGS if s in set(comp["setting"])]
    summ = summ.reindex(columns=[(m, s) for m in METRICS for s in present])
    return summ.reindex(model_order(comp))


def available_comparisons(comp, baseline=BASELINE):
    """A do artigo (WP + CP) sempre; a realista (WP + CP_crono) se o baseline rodou no CP_crono."""
    has_crono = ((comp["model"] == baseline) & (comp["setting"] == "CP_crono")).any()
    return [ARTIGO, REALISTA] if has_crono else [ARTIGO]


def compare_to_baseline(comp, metric, baseline=BASELINE, settings=ARTIGO):
    """Wilcoxon pareado + Cliff's delta de cada modelo contra o baseline numa metrica.

    Os pares sao projeto x cenario, so dos cenarios em `settings` (12 com dois cenarios). O
    p-valor e corrigido com Holm entre os modelos comparados na mesma metrica.
    """
    comp = comp[comp["setting"].isin(settings)]
    wide = comp.pivot_table(index=["project", "setting"], columns="model", values=metric,
                            dropna=False)
    if baseline not in wide.columns:
        raise KeyError(f"baseline {baseline} sem resultados de {metric}")
    rows = []
    for model in model_order(comp, baseline):
        if model == baseline or model not in wide.columns:
            continue
        # o modelo precisa ter a metrica em todos os cenarios da comparacao. Ex.: os modelos do
        # artigo nao tem CP_crono, e no realista entrariam so com os 6 pares do WP
        has = wide[model].notna().groupby(level="setting").any()
        if not all(has.get(s, False) for s in settings):
            continue
        r = paired_test(wide[model].to_numpy(), wide[baseline].to_numpy())
        rows.append({"metric": metric, "model": model, "baseline": baseline, **r})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_value"])
    return out


def verdict(row, alpha=ALPHA):
    # so conta como diferenca real se for significativa depois do Holm e o efeito nao for negligivel
    if pd.isna(row["p_holm"]) or row["p_holm"] >= alpha or row["magnitude"] == "negligível":
        return "empate"
    return "supera" if row["delta"] > 0 else "perde"


def all_tests(comp, baseline=BASELINE, settings=ARTIGO):
    # so testa as metricas que o baseline tem (ex.: CSV antigo, sem as metricas de esforco)
    has = comp.loc[(comp["model"] == baseline) & comp["setting"].isin(settings),
                   METRICS].notna().any()
    tests = [compare_to_baseline(comp, m, baseline, settings) for m in METRICS if has[m]]
    out = pd.concat([t for t in tests if not t.empty], ignore_index=True)
    out["verdict"] = out.apply(verdict, axis=1)
    out["cenarios"] = "+".join(settings)
    return out


def answer(tests, baseline=BASELINE, alpha=ALPHA):
    """Responde em texto: algum modelo supera o baseline? (titulo, linha vazia, resposta...)"""
    winners = tests[tests["verdict"] == "supera"]
    realista = "cenarios" in tests and (tests["cenarios"] == "+".join(REALISTA)).all()
    title = (f"## Algum modelo supera o {baseline} no cenário realista (WP + CP cronológico)?"
             if realista else f"## Algum modelo supera o {baseline}?")
    lines = [title, ""]
    if winners.empty:
        lines.append(f"**Não.** Nenhum modelo foi significativamente melhor que o {baseline} "
                     f"em nenhuma métrica (Wilcoxon pareado com Holm, alfa = {alpha}, "
                     "e Cliff's delta não negligível).")
    else:
        found = ", ".join(f"{r.model} em {METRIC_NAMES[r.metric]}" for r in winners.itertuples())
        lines.append(f"**Sim:** {found}.")
    lines += ["", "| Modelo | Métrica | Pares | Vitórias/derrotas | Mediana da diferença "
              "| p (Holm) | Cliff's delta | Resultado |", "|---|---|---|---|---|---|---|---|"]
    for r in tests.itertuples():
        lines.append(f"| {r.model} | {METRIC_NAMES[r.metric]} | {r.n} | {r.wins}/{r.losses} "
                     f"| {r.median_diff:+.4f} | {r.p_holm:.4f} | {r.delta:+.3f} ({r.magnitude}) "
                     f"| {r.verdict} |")
    return "\n".join(lines) + "\n"


REPORT = ROOT / "results" / "comparacao.md"


def summary_markdown(summ):
    # sem tabulate: monta a tabela na mao
    cols = [f"{METRIC_NAMES[m]} {s}" for m, s in summ.columns]
    lines = ["| Modelo | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for model, row in summ.iterrows():
        vals = ["—" if pd.isna(v) else f"{v:.3f}" for v in row]
        lines.append(f"| {model} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _effort_reference_note(path):
    """Nota com a referencia sem modelo (src/evaluation/effort_baseline.py), se o CSV existir."""
    if not path.exists():
        return []
    ref = pd.read_csv(path).groupby("setting")[["popt", "recall_20"]].mean()
    ref = ref.reindex([s for s in ("WP", "CP", "CP_crono") if s in ref.index])
    values = "; ".join(f"{r.Index}: Popt {r.popt:.3f}, Recall@20% {r.recall_20:.3f}"
                       for r in ref.itertuples())
    return ["**Referência sem modelo para Popt e Recall@20%:** revisar do menor commit para o "
            f"maior, sem usar nenhum modelo, já dá {values}. Como a ordem de revisão é a "
            "densidade de risco (probabilidade ÷ linhas), o tamanho do commit pesa muito nessas "
            "métricas, e o mérito de um modelo é só o quanto ele passa dessa referência "
            "(ver Yang et al., 2016).", ""]


def save_outputs(comp, tests, metrics_dir=METRICS_DIR, report=REPORT):
    """Salva as tabelas em results/metrics/ e o relatorio em results/comparacao.md."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    order = model_order(comp)
    long = comp.assign(model=pd.Categorical(comp["model"], order)).sort_values(KEYS)
    path = metrics_dir / "comparacao_resultados.csv"
    long.to_csv(path, index=False)
    paths.append(path)
    for metric in METRICS:
        path = metrics_dir / f"comparacao_{metric}.csv"
        metric_table(comp, metric).to_csv(path)
        paths.append(path)
    summ = summary_table(comp)
    path = metrics_dir / "comparacao_resumo.csv"
    summ.to_csv(path)
    paths.append(path)
    path = metrics_dir / "comparacao_testes.csv"
    tests.to_csv(path, index=False)
    paths.append(path)

    paper = sorted(set(comp.loc[comp["source"] == "artigo", "model"]))
    deep = sorted(set(comp.loc[comp["source"] == "autores", "model"]))
    text = ["# Comparação final", "",
            "Gerado por `python -m src.evaluation.compare`. Não edite à mão.", "",
            "## Média dos 6 projetos", "", summary_markdown(summ), ""]
    if deep:
        text += [f"{', '.join(deep)}: AUC e PR-AUC calculados a partir das predições publicadas "
                 "pelos autores. F1 e MCC ficam de fora porque não há dados de validação "
                 "para escolher o limiar, e Popt e Recall@20% porque as predições não foram "
                 "cruzadas com as linhas de cada commit.", ""]
    if paper:
        text += [f"{', '.join(paper)}: só a AUC publicada no artigo (Tabela 8).", ""]
    text += _effort_reference_note(metrics_dir / "esforco_referencia.csv")
    groups = tests["cenarios"] if "cenarios" in tests else pd.Series("+".join(ARTIGO),
                                                                      index=tests.index)
    text += [answer(tests[groups == "+".join(ARTIGO)]),
             "Pares = projeto × cenário (WP e CP), com a média das seeds de cada modelo. "
             "O p-valor é do Wilcoxon pareado bicaudal, corrigido por Holm entre os modelos "
             "de cada métrica. Magnitude do Cliff's delta pelos limiares de Romano et al. (2006).",
             "",
             "**Limitação do CP:** o treino usa os outros cinco projetos inteiros, então parte "
             "dos commits de treino é mais recente que o período de teste. O artigo faz assim "
             "e mantivemos para comparar nas mesmas condições. A comparação abaixo corrige isso.",
             ""]
    realista = tests[groups == "+".join(REALISTA)]
    if not realista.empty:
        text += [answer(realista),
                 "Mesmo teste, trocando o CP pelo CP_crono: o mesmo período de teste, mas o "
                 "treino só tem os commits dos outros projetos anteriores ao primeiro commit de "
                 "teste, como seria num uso real. Os pares continuam 12 (WP e CP_crono), porque "
                 "com só os 6 do CP_crono o Wilcoxon não chega a p < 0,05 depois do Holm.",
                 ""]
    text += ["Figuras em `results/figures/`.", ""]
    report.write_text("\n".join(text), encoding="utf-8")
    paths.append(report)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args()

    comp = build_comparison(load_results())
    order = model_order(comp, args.baseline)
    by_comparison = {s: all_tests(comp, args.baseline, s)
                     for s in available_comparisons(comp, args.baseline)}
    tests = pd.concat(by_comparison.values(), ignore_index=True)
    paths = save_outputs(comp, tests)
    paths += [plots.plot_metrics(comp, order, args.baseline),
              plots.plot_auc_by_project(comp, order),
              plots.plot_effect_sizes(by_comparison[ARTIGO], args.baseline)]
    if REALISTA in by_comparison:
        paths.append(plots.plot_effect_sizes(
            by_comparison[REALISTA], args.baseline,
            path=plots.FIGURES_DIR / "cliffs_delta_lapredict_realista.png",
            scenario="WP + CP cronológico"))

    print(summary_table(comp).round(3).to_string(), "\n")
    for t in by_comparison.values():
        print(answer(t, args.baseline))
    print("Salvo em:\n" + "\n".join(f"  {p.relative_to(ROOT)}" for p in paths))


if __name__ == "__main__":
    main()
