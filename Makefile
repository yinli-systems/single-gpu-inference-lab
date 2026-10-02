ARTIFACT_PYTHON ?= python3
ARTIFACT_OUT ?= artifact/generated

.PHONY: reproduce-v42-paper
reproduce-v42-paper:
	$(ARTIFACT_PYTHON) -m research.selector_v4.system_artifact.bootstrap --out $(ARTIFACT_OUT)
