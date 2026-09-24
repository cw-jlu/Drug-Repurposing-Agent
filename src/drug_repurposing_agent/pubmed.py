"""PubMed retrieval for human review; search hits are never evidence grades."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from time import monotonic, sleep
from urllib.parse import urlencode
from urllib.request import urlopen


BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


class PubMedClient:
    def __init__(self, min_interval_seconds: float = 0.36):
        self.min_interval_seconds = min_interval_seconds
        self._last_request = 0.0

    def _get(self, endpoint: str, **params: str | int) -> dict:
        remaining = self.min_interval_seconds - (monotonic() - self._last_request)
        if remaining > 0:
            sleep(remaining)
        query = urlencode({"db": "pubmed", "retmode": "json",
                           "tool": "DrugRepurposingAgent", **params})
        with urlopen(BASE + endpoint + "?" + query, timeout=30) as response:
            result = json.load(response)
        self._last_request = monotonic()
        return result

    def search_luad(self, drug_name: str, retmax: int = 5) -> dict:
        if not drug_name or retmax < 1 or retmax > 20:
            raise ValueError("Drug name and retmax 1-20 required")
        query = (f'"{drug_name}"[Title/Abstract] AND '
                 '("lung adenocarcinoma"[Title/Abstract] OR '
                 '"non-small cell lung cancer"[Title/Abstract])')
        search = self._get("esearch.fcgi", term=query, retmax=retmax,
                           sort="relevance")["esearchresult"]
        ids = search["idlist"]
        records = []
        if ids:
            summaries = self._get("esummary.fcgi", id=",".join(ids))["result"]
            for pmid in ids:
                summary = summaries[pmid]
                doi = next((item["value"] for item in summary.get("articleids", [])
                            if item.get("idtype") == "doi"), None)
                records.append({"pmid": pmid, "doi": doi,
                                "title": summary.get("title"),
                                "publication_date": summary.get("pubdate"),
                                "journal": summary.get("source"),
                                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                "review_status": "unreviewed_search_hit"})
        return {"drug_name": drug_name, "query": query,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "total_hits": int(search["count"]), "records": records,
                "interpretation": "Search hits require article-level review; no efficacy inference."}
