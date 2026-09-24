from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_array

from drug_repurposing_agent.benchmark import TranscriptBaseline
from drug_repurposing_agent.data import ExpressionData
from drug_repurposing_agent.differential_expression import paired_deg
from drug_repurposing_agent.evidence import CandidateLedger, Citation
from drug_repurposing_agent.pubmed import PubMedClient
from drug_repurposing_agent.ranking import _rrf, connectivity_gene_sets, score_expressions
from drug_repurposing_agent.workflow import run_expression_workflow


def example_data():
    genes = ["g1", "g2", "g3", "g4", "g5", "g6"]
    drugs = pd.DataFrame({"reverse": [-3, -2, -1, 1, 2, 3],
                          "same": [3, 2, 1, -1, -2, -3]}, index=genes)
    diseases = pd.DataFrame({"disease": [3, 2, 1, -1, -2, -3]}, index=genes).iloc[::-1]
    return ExpressionData(drugs, diseases)


def test_explicit_gene_alignment_and_reversal():
    data = example_data()
    scores = score_expressions(data, top_k=2)
    # The disease gene order was reversed before constructing ExpressionData.
    assert list(data.diseases.index) == list(data.drugs.index)
    assert scores["spearman_reversal"].loc["reverse", "disease"] > 0
    assert scores["spearman_reversal"].loc["same", "disease"] < 0
    assert scores["gene_counts"].iloc[0, 0] == 6


def test_missing_and_constant_vectors_have_finite_scores():
    data = example_data()
    data.drugs.loc["g1", "reverse"] = np.nan
    data.drugs["constant"] = 1.0
    scores = score_expressions(data, top_k=2)
    assert np.isfinite(scores["spearman_reversal"].to_numpy()).all()
    assert scores["gene_counts"].loc["reverse", "disease"] == 5
    assert scores["spearman_reversal"].loc["constant", "disease"] == 0


def test_constant_disease_is_unrankable():
    data = example_data()
    data.diseases["flat"] = 1.0
    scores = score_expressions(data, top_k=2)
    assert (scores["spearman_reversal"]["flat"] == 0).all()
    assert (scores["connectivity"]["flat"] == 0).all()
    assert (scores["rrf"]["flat"] == 0).all()


def test_prespecified_gene_set_connectivity_prefers_reversal():
    data = example_data()
    values = connectivity_gene_sets(data.drugs, {"g1", "g2"}, {"g5", "g6"})
    assert values["reverse"] > values["same"]
    with pytest.raises(ValueError, match="disjoint"):
        connectivity_gene_sets(data.drugs, {"g1"}, {"g1"})


def test_rrf_ties_do_not_depend_on_input_order():
    scores = np.array([[0.0], [1.0], [0.0], [1.0]])
    original = _rrf([scores])[:, 0]
    order = np.array([3, 2, 1, 0])
    reordered = _rrf([scores[order]])[:, 0]
    assert np.allclose(original[order], reordered)
    assert original[0] == original[2]
    assert original[1] == original[3]


def test_workflow_never_needs_ratings_and_writes_provenance(tmp_path: Path):
    data = example_data()
    items, users = tmp_path / "items.csv", tmp_path / "users.csv"
    data.drugs.to_csv(items)
    data.diseases.to_csv(users)
    manifest = run_expression_workflow(items, users, tmp_path / "out", top_k=2)
    assert manifest["qc"]["genes"] == 6
    assert len(manifest["input"]["items"]["sha256"]) == 64
    assert (tmp_path / "out" / "rrf.csv").exists()


class FakeDataset:
    def __init__(self, data, labels=None):
        self.item_list = list(data.drugs.columns)
        self.user_list = list(data.diseases.columns)
        self.item_features = list(data.drugs.index)
        self.user_features = list(data.diseases.index)
        self.items = coo_array(data.drugs.to_numpy())
        self.users = coo_array(data.diseases.to_numpy())
        self.folds = coo_array(np.ones((2, 1)))
        self._ratings = labels

    @property
    def ratings(self):
        if self._ratings is None:
            raise AssertionError("Validation labels were accessed")
        return coo_array(self._ratings)


@pytest.mark.parametrize("method", ["B0", "B0p", "B1", "B1k", "B2"])
def test_benchmark_adapter_does_not_read_validation_labels(method):
    data = example_data()
    train = FakeDataset(data, np.array([[1], [0]]))
    validation = FakeDataset(data)
    model = TranscriptBaseline({"method": method}).fit(train)
    result = model.predict_proba(validation)
    assert result.shape == (2, 1)
    assert result.nnz == 2
    assert np.isfinite(result.data).all()


def test_evidence_citation_validation():
    ledger = CandidateLedger("DB1", "C1", "trace", 1, {"reversal": 0.8},
                             supporting_evidence=[Citation("PMID", "12345678", "support", "")])
    ledger.validate()
    ledger.supporting_evidence = [Citation("PMID", "made-up", "support", "")]
    with pytest.raises(ValueError):
        ledger.validate()


def test_pubmed_hits_remain_unreviewed(monkeypatch):
    client = PubMedClient()
    def fake_get(endpoint, **params):
        if endpoint == "esearch.fcgi":
            assert '"drug-X"[Title/Abstract]' in params["term"]
            return {"esearchresult": {"count": "1", "idlist": ["12345"]}}
        return {"result": {"12345": {"title": "A study", "pubdate": "2020",
                                     "source": "Journal", "articleids": []}}}
    monkeypatch.setattr(client, "_get", fake_get)
    result = client.search_luad("drug-X")
    assert result["total_hits"] == 1
    assert result["records"][0]["review_status"] == "unreviewed_search_hit"
    assert "efficacy" in result["interpretation"]


def test_paired_deg_matches_patient_pairs_and_rejects_missing_pair():
    samples = pd.DataFrame({"sample_id": ["n2", "t1", "n1", "t3", "t2", "n3"],
                            "patient_id": ["2", "1", "1", "3", "2", "3"],
                            "condition": ["Normal", "Tumor", "Normal", "Tumor", "Tumor", "Normal"]})
    expression = pd.DataFrame({"n2": [2, 8], "t1": [4, 7], "n1": [1, 8],
                               "t3": [5, 7.5], "t2": [5, 6.5], "n3": [1, 8]}, index=["G1", "G2"])
    result = paired_deg(expression, samples)
    assert result.loc[0, "log2FC"] == pytest.approx(10 / 3)
    assert result.loc[1, "log2FC"] == -1
    assert result.loc[0, "n_pairs"] == 3
    with pytest.raises(ValueError, match="exactly one"):
        paired_deg(expression.drop(columns="n3"), samples.iloc[:-1])
