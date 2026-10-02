ARTIFACT_PYTHON ?= python3
ARTIFACT_OUT ?= artifact/generated
GRAPH_HTTP_OUT ?= artifact/generated-graph-http

.PHONY: reproduce-v42-paper
reproduce-v42-paper:
	$(ARTIFACT_PYTHON) -m research.selector_v4.system_artifact.bootstrap --out $(ARTIFACT_OUT)

.PHONY: reproduce-graph-http
reproduce-graph-http:
	artifact/.venv/bin/python artifact/graph_http_artifact.py --output $(GRAPH_HTTP_OUT)
