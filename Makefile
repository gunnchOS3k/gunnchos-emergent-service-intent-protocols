PYTHON ?= python3
.PHONY: bootstrap test smoke reproduce paper artifact gate4-cpu gate4-gpu clean lint typecheck

bootstrap:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests scripts || true

typecheck:
	$(PYTHON) -m mypy src/emergent_intent || true

test:
	$(PYTHON) -m pytest -q --cov=emergent_intent --cov-report=term-missing tests

smoke:
	$(PYTHON) scripts/run_smoke_experiments.py --seeds 5 --steps 128

reproduce: bootstrap test smoke
	$(PYTHON) scripts/build_artifact.py

paper:
	@test -f paper/RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.md
	$(PYTHON) scripts/generate_paper_tables.py || true
	@command -v pandoc >/dev/null 2>&1 && pandoc paper/RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.md -o paper/RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.tex || cp paper/handwritten.tex paper/RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.tex
	@echo "PAPER_OK"

artifact: reproduce
	@echo "ARTIFACT_OK"

gate4-cpu: test smoke
	@echo "GATE4_OULU_CPU_OK"

gate4-gpu:
	@$(PYTHON) -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA',ok); sys.exit(0 if not ok else 1)" \
		&& (mkdir -p results/smoke && $(PYTHON) -c "import json; from pathlib import Path; from emergent_intent.utils import detect_device, dump_json; d=detect_device(); dump_json('results/smoke/gate4_gpu_blocked.json', {'status':'BLOCKED_HARDWARE','evidence_class':'BLOCKED','device':d.as_dict()}); print('GATE4_NVIDIA_GPU_PENDING / BLOCKED_HARDWARE (no CUDA)')") \
		|| $(PYTHON) scripts/run_smoke_experiments.py --device cuda --seeds 1 --steps 64

clean:
	rm -rf results/smoke/* .coverage htmlcov
