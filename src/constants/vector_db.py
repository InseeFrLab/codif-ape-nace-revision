# Tuned for vLLM: too much client-side concurrency causes timeouts since vLLM batches requests internally.
MAX_CONCURRENCY = 32
