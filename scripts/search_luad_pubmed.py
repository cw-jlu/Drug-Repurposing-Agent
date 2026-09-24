"""Retrieve PubMed leads for the frozen Top-10 without automatic promotion."""

from pathlib import Path
import json

import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.pubmed import PubMedClient


if __name__ == "__main__":
    root = Path("artifacts/reports/luad_eh3226")
    ranking_path = root / "all_candidates.csv"
    names = pd.read_csv(ranking_path).head(10).drug_name.tolist()
    client = PubMedClient()
    results = [client.search_luad(name) for name in names]
    output = {"source": "NCBI PubMed E-utilities ESearch/ESummary",
              "api_docs": "https://www.ncbi.nlm.nih.gov/books/NBK25499/",
              "ranking_sha256": sha256_file(ranking_path),
              "status": "retrieval_only_unreviewed", "queries": results}
    (root / "pubmed_search.json").write_text(json.dumps(output, indent=2,
                                                        ensure_ascii=False), encoding="utf-8")
    print(json.dumps({entry["drug_name"]: entry["total_hits"] for entry in results}, indent=2))
