"""Comparacao final (#7): junta os resultados de todos os modelos e compara com o LApredict.

Le todos os results/metrics/experimentos_*.csv, entao modelo novo entra sozinho.

Uso: python -m src.evaluation.compare
Saida: results/metrics/comparacao_*.csv, results/figures/*.png e results/comparacao.md
"""
import argparse

import pandas as pd

from src.data.loader import PROJECTS, ROOT
from src.evaluation import plots
from src.evaluation.metrics import METRIC_NAMES
from src.evaluation.reproduce import REFERENCE, load_reference
from src.evaluation.stats import holm, paired_test

METRICS_DIR = ROOT / "results" / "metrics"
METRICS = ["auc", "pr_auc", "f1", "mcc"]
KEYS = ["model", "project", "setting"]
BASELINE = "lapredict"
ALPHA = 0.05
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
    agg = results.groupby(KEYS, as_index=False)[METRICS].mean()
    agg["source"] = "nosso"
    return agg


def paper_results(skip=(), path=REFERENCE):
    # o artigo so publica AUC, o resto fica NaN ate a #6 calcular das predicoes dos autores
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


def build_comparison(results, path=REFERENCE):
    ours = aggregate(results)
    # se alguem rodar o modelo de verdade (ex: #6), o nosso substitui o do artigo
    paper = paper_results(skip=set(ours["model"]), path=path)
    return pd.concat([ours, paper], ignore_index=True)


def model_order(comp, baseline="lapredict"):
    # baseline primeiro, depois por AUC media decrescente
    mean_auc = comp.groupby("model")["auc"].mean().sort_values(ascending=False)
    others = [m for m in mean_auc.index if m != baseline]
    return ([baseline] if baseline in mean_auc.index else []) + others


def metric_table(comp, metric):
    """Uma linha por modelo, uma coluna por projeto x cenario, mais a media de WP e de CP."""
    table = comp.pivot_table(index="model", columns=["setting", "project"], values=metric,
                             dropna=False)
    table = table.dropna(how="all")
    for setting in ("WP", "CP"):
        if setting in table.columns.get_level_values(0):
            table[(setting, "media")] = table[setting].mean(axis=1, skipna=False)
    table = table.reindex(columns=sorted(table.columns, key=_column_key))
    return table.reindex([m for m in model_order(comp) if m in table.index])


def _column_key(col):
    setting, project = col
    order = PROJECTS + ["media"]
    return (["WP", "CP"].index(setting), order.index(project) if project in order else 99)


def summary_table(comp):
    """Media das 6 bases por modelo e cenario, com as 4 metricas lado a lado."""
    summ = comp.groupby(["model", "setting"])[METRICS].mean().unstack("setting")
    summ = summ.reindex(columns=[(m, s) for m in METRICS for s in ("WP", "CP")])
    return summ.reindex(model_order(comp))


def compare_to_baseline(comp, metric, baseline=BASELINE):
    """Wilcoxon pareado + Cliff's delta de cada modelo contra o baseline numa metrica.

    Os pares sao projeto x cenario (12 quando o modelo tem todos). O p-valor e corrigido
    com Holm entre os modelos comparados na mesma metrica.
    """
    wide = comp.pivot_table(index=["project", "setting"], columns="model", values=metric,
                            dropna=False)
    if baseline not in wide.columns:
        raise KeyError(f"baseline {baseline} sem resultados de {metric}")
    rows = []
    for model in model_order(comp, baseline):
        if model == baseline or model not in wide.columns or wide[model].isna().all():
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


def all_tests(comp, baseline=BASELINE):
    tests = [compare_to_baseline(comp, m, baseline) for m in METRICS]
    out = pd.concat([t for t in tests if not t.empty], ignore_index=True)
    out["verdict"] = out.apply(verdict, axis=1)
    return out


def answer(tests, baseline=BASELINE, alpha=ALPHA):
    """Responde em texto: algum modelo supera o baseline?"""
    winners = tests[tests["verdict"] == "supera"]
    lines = [f"## Algum modelo supera o {baseline}?", ""]
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
    text = ["# Comparação final", "",
            "Gerado por `python -m src.evaluation.compare`. Não edite à mão.", "",
            "## Média dos 6 projetos", "", summary_markdown(summ), ""]
    if paper:
        text += [f"{', '.join(paper)}: só a AUC publicada no artigo (Tabela 8); "
                 "as outras métricas dependem da #6.", ""]
    text += [answer(tests),
             "Pares = projeto × cenário, com a média das seeds de cada modelo. "
             "O p-valor é do Wilcoxon pareado bicaudal, corrigido por Holm entre os modelos "
             "de cada métrica. Magnitude do Cliff's delta pelos limiares de Romano et al. (2006).",
             "", "Figuras em `results/figures/`.", ""]
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
    tests = all_tests(comp, args.baseline)
    paths = save_outputs(comp, tests)
    paths += [plots.plot_metrics(comp, order, args.baseline),
              plots.plot_auc_by_project(comp, order),
              plots.plot_effect_sizes(tests, args.baseline)]

    print(summary_table(comp).round(3).to_string(), "\n")
    print(answer(tests, args.baseline))
    print("Salvo em:\n" + "\n".join(f"  {p.relative_to(ROOT)}" for p in paths))


if __name__ == "__main__":
    main()
