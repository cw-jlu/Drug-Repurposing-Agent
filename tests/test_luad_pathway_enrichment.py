from scripts.luad_pathway_enrichment import ora


def test_ora_detects_planted_enrichment_and_skips_small_sets():
    background = {f"G{i}" for i in range(1000)}
    library = {"planted": [f"G{i}" for i in range(50)],
               "null": [f"G{i}" for i in range(500, 550)],
               "tiny": ["G1", "G2"]}
    selected = {f"G{i}" for i in range(40)} | {f"G{i}" for i in range(900, 960)}
    rows = {r["term"]: r for r in ora(selected, background, library)}
    assert "tiny" not in rows
    assert rows["planted"]["overlap"] == 40 and rows["planted"]["fdr"] < 1e-10
    assert rows["null"]["overlap"] == 0 and rows["null"]["p_value"] == 1.0
