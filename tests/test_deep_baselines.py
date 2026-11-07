from pathlib import Path

import pandas as pd
import pytest

REF = Path("data/reference/rq4_t8.csv")
OUT = Path("results/metrics/deep_baselines.csv")


def load_reference():
    ref = pd.read_csv(REF, header=None, dtype=str).apply(lambda c: c.str.strip())
    projects, settings = ref.iloc[0, 1:13], ref.iloc[1, 1:13]
    rows = []
    for _, r in ref.iloc[2:4].iterrows():
        for col, (p, s) in enumerate(zip(projects, settings), start=1):
            rows.append({"model": r[0].lower(), "project": p, "setting": s,
                         "auc_ref": float(r[col])})
    return pd.DataFrame(rows)


@pytest.mark.skipif(not OUT.exists(), reason="rode deep_baselines antes")
def test_auc_bate_com_tabela_8():
    mine = pd.read_csv(OUT)
    merged = mine.merge(load_reference(), on=["model", "project", "setting"],
                        how="outer", indicator=True)
    assert (merged["_merge"] == "both").all()
    assert (merged["auc"] - merged["auc_ref"]).abs().max() < 1e-3

@pytest.mark.skipif(not OUT.exists(), reason="rode deep_baselines antes")
def test_csv_tem_formato_esperado():
    df = pd.read_csv(OUT)
    assert list(df.columns) == ["model", "project", "setting", "auc", "pr_auc"]
    assert len(df) == 24
    assert set(df["model"]) == {"deepjit", "cc2vec"}
    assert set(df["setting"]) == {"WP", "CP"}
    assert df[["auc", "pr_auc"]].notna().all().all()