# Runbook

| Alert | First checks | Action |
|---|---|---|
| `issue_spike` | Which merchant / channel dominates the week? Any incident ticket? | Notify the product owner; add a holding reply template |
| `unclassified_surge` / emerging topic | Read the topic examples in the console | Decide: new taxonomy label or existing one; label 100 examples; retrain |
| Drift: `vocabulary` / `novelty` | New channel or language mix? | Run a review cycle; retrain with `make train` |
| Drift: `issue_mix` | Real change in complaints or model error? | Compare with manual sample before acting |
| PII leak reported | Reproduce with `/v1/redact` | Add a failing test, fix the rule, purge affected logs |
| Reply drafts all `template_fallback` | LLM unavailable or guard rejecting | Check `/ready`; templates are safe to keep using |
