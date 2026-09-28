# Trace requirement

For every future Agent run, model request, evaluation, or research workflow in this repository, record a durable trace by default. Create the trace before the operation, append events as they happen, record validation decisions and failures, and include the trace path in the result or handoff. Do not report a run as verified without checking its trace exists.

Use `drug_repurposing_agent.trace.TraceRecorder` for new Python entrypoints. Keep traces under ignored `artifacts/` or the run's output directory, with unique files rather than overwriting prior runs. Never put API keys, authorization headers, credentials, or model-internal reasoning in a trace. Treat prompts, responses, and biomedical data in traces as potentially sensitive; do not commit them without deliberate review. Historical results without raw traces must remain labeled as such; do not reconstruct or imply unavailable model reasoning.
