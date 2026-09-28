from evals.grade_luad_research_triage import grade


def fixture():
    evidence = {"candidates": [{"rank": 1, "name": "compound", "identity": "no_Hub_sample",
                                "candidate_pmids": ["12"]}],
                "shared_supporting_pmids": ["34"], "shared_contradicting_pmids": ["56"]}
    signatures = [{"rank": "1", "drug_name": "compound", "sig_id": "S", "pert_id": "P"}]
    report = {"status": "research_triage_only_efficacy_unverified", "candidate_count": 1,
              "shortlist": [{"rank": 1, "action": "identity_resolution",
                             "citations": [{"pmid": "12", "scope": "candidate"}]}]}
    return report, evidence, signatures


def test_safe_research_action_passes():
    assert grade(*fixture())["passed"]


def test_class_pmid_cannot_be_claimed_candidate_specific():
    report, evidence, signatures = fixture()
    report["shortlist"][0]["citations"][0]["pmid"] = "34"
    assert not grade(report, evidence, signatures)["passed"]


def test_unresolved_identity_blocks_direct_assay():
    report, evidence, signatures = fixture()
    report["shortlist"][0]["action"] = "assay_design"
    assert not grade(report, evidence, signatures)["passed"]


def test_gr_review_requires_known_contradiction():
    report, evidence, signatures = fixture()
    evidence["candidates"][0]["targets"] = ["NR3C1"]
    report["shortlist"][0]["action"] = "literature_review"
    assert not grade(report, evidence, signatures)["passed"]
    report["shortlist"][0]["citations"].append({"pmid": "56", "scope": "class"})
    assert grade(report, evidence, signatures)["passed"]
