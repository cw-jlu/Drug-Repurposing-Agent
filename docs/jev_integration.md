# Optional Jev decision adapter

The adapter in `src/drug_repurposing_agent/jev.py` follows the [TypeSafe System One OpenAPI](https://api.typesafe.ai/docs): `POST /v1/systemone` accepts a structured state and named Choice, Score, and Noul questions; `GET /v1/models` returns model names available to an authenticated account. The API key is read only from `TYPESAFE_API_KEY` and is never stored in reports. No live TypeSafe account or API result has been used for the project results so far.

The client checks that every returned answer has the requested type, declared choices or score levels, finite probabilities, and a valid probability distribution. A separate deterministic gate abstains to `manual_review` when the provider is unavailable or confidence is below 0.8 (0.9 for high-risk choices). A Jev choice to `publish_clinical` always goes to manual review. These are project policy thresholds for safe routing; they are not calibrated or validated Jev performance results. Source hashes, candidate IDs, citation relevance, clinical claims, and benchmark label isolation remain code-level checks.

Example of the *shape* of an optional call (requires a valid TypeSafe account and explicit execution):

```python
from drug_repurposing_agent.jev import JevClient, choice_question, gate_choice

client = JevClient.from_env()
question = choice_question("Route this QC record", {
    "continue": "All required inputs and checks are present",
    "manual_review": "A required input or check is uncertain",
})
response = client.ask({"pairs": 57, "excluded_samples": 2}, {"route": question})
decision = gate_choice(question, response["answers"]["route"], high_risk=True)
```

The current LUAD case runs fully with deterministic rules and records zero external-model calls. `python scripts/build_luad_case.py --use-jev` explicitly enables one advisory research-task routing question when `TYPESAFE_API_KEY` is present. Any API or schema failure falls back to manual review; the case can never be promoted to clinical evidence by this choice. A live Jev-versus-rule/LLM comparison requires API credentials, a frozen decision-eval set with reference labels, and measured usage, latency, calibration, and error rates. No such comparison is claimed.
