PYTHON ?= python3
.PHONY: bootstrap test smoke causal-tests algorithm-validation final-experiments statistics figures paper artifact reproduce-clean gate4-cpu gate4-gpu clean lint typecheck

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

causal-tests:
	$(PYTHON) -m pytest -q tests/unit/test_observation_inbox.py tests/unit/test_causal_actions.py tests/unit/test_comm_necessary_scenarios.py

algorithm-validation:
	$(PYTHON) -m pytest -q tests/unit/test_dial_tarmac_fidelity.py tests/unit/test_qmix_vdn.py tests/unit/test_ippo_mappo.py tests/unit/test_status_integrity.py

final-experiments:
	@mkdir -p results/smoke results/pilot results/final results/ablations results/generalization results/robustness results/interpretability
	$(PYTHON) scripts/run_pilot_experiments.py --seeds 5 --steps 64
	@echo "NOTE: outputs are labeled PILOT; results/final/STATUS.json remains NOT_RUN for short budgets."

statistics:
	$(PYTHON) scripts/run_statistics.py

figures:
	$(PYTHON) scripts/generate_figures.py
	$(PYTHON) scripts/generate_paper_tables.py || true

paper: figures
	$(PYTHON) scripts/validate_paper.py
	@command -v pandoc >/dev/null 2>&1 && pandoc paper/RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.md -o paper/RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.tex || cp paper/handwritten.tex paper/RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.tex
	@echo "PAPER_BUILD_OK"

artifact: reproduce-clean
	$(PYTHON) scripts/build_artifact.py
	@echo "ARTIFACT_OK"

reproduce-clean: bootstrap test causal-tests algorithm-validation
	$(PYTHON) scripts/run_pilot_experiments.py --quick --steps 32
	$(PYTHON) scripts/run_statistics.py
	$(PYTHON) scripts/generate_figures.py
	$(PYTHON) scripts/validate_paper.py

gate4-cpu: test smoke causal-tests algorithm-validation
	@echo "GATE4_OULU_CPU_OK (smoke≠scientific final)"

gate4-gpu:
	@$(PYTHON) -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA',ok); sys.exit(0 if not ok else 1)" \
		&& (mkdir -p results/smoke && $(PYTHON) -c "import json; from pathlib import Path; from emergent_intent.utils import detect_device, dump_json; d=detect_device(); dump_json('results/smoke/gate4_gpu_blocked.json', {'status':'BLOCKED_HARDWARE','evidence_class':'BLOCKED','device':d.as_dict()}); print('GATE4_NVIDIA_GPU_PENDING / BLOCKED_HARDWARE (no CUDA)')") \
		|| $(PYTHON) scripts/run_smoke_experiments.py --device cuda --seeds 1 --steps 64

clean:
	rm -rf results/smoke/* results/pilot/* .coverage htmlcov
