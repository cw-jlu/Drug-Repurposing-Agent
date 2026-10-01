import numpy as np
import pandas as pd

from scripts.luad_pathway_reversal import pathway_reversal


def test_reversal_sign_and_min_gene_rule():
    genes = [f"G{i}" for i in range(12)]
    deg = pd.DataFrame({"log2FC": [2.0] * 6 + [-2.0] * 6, "fdr": [0.01] * 12}, index=genes)
    drugs = pd.DataFrame({"reverser": [-1.0] * 6 + [1.0] * 6, "mimic": [1.0] * 6 + [-1.0] * 6},
                         index=genes)
    library = {"UP_SET": genes[:6], "DOWN_SET": genes[6:], "SMALL": genes[:3]}
    scores, used = pathway_reversal(drugs, deg, library, {"up": ["UP_SET", "SMALL"], "down": ["DOWN_SET"]})
    assert len(used) == 2  # SMALL has < 5 genes
    assert (scores.loc["reverser"] > 0).all() and (scores.loc["mimic"] < 0).all()
