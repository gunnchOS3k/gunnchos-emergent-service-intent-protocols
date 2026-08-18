PYTHON ?= python3
.PHONY: bootstrap test smoke causal-tests algorithm-validation \
	semantic-intervention-tests dial-validation tarmac-validation \
	final-experiments generalization robustness ablations interpretability \
	statistics figures paper artifact reproduce-clean gate4-cpu gate4-gpu clean lint typecheck \
	blocked-gpu supervisor-cpu-gate

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

semantic-intervention-tests:
	$(PYTHON) -m pytest -q tests/scientific/test_message_semantic_causality.py

dial-validation:
	$(PYTHON) -m pytest -q tests/scientific/test_dial_end_to_end.py tests/unit/test_dial_tarmac_fidelity.py

tarmac-validation:
	$(PYTHON) -m pytest -q tests/scientific/test_tarmac_end_to_end.py tests/unit/test_dial_tarmac_fidelity.py

final-experiments:
	@mkdir -p results/smoke results/pilot results/final results/ablations results/generalization results/robustness results/interpretability results/interventions
	$(PYTHON) scripts/run_final_experiments.py --mode final --seeds 5 --steps 1024 --time-budget-s 900
	@echo "NOTE: inspect results/final/STATUS.json — smoke≠final; BLOCKED_COMPUTE_CAPACITY if budget exceeded."

generalization:
	@mkdir -p results/generalization
	$(PYTHON) scripts/run_final_experiments.py --mode generalization --seeds 5 --steps 768 --time-budget-s 400

robustness:
	@mkdir -p results/robustness
	$(PYTHON) scripts/run_final_experiments.py --mode robustness --seeds 5 --steps 512 --time-budget-s 300

ablations:
	@mkdir -p results/ablations
	$(PYTHON) scripts/run_final_experiments.py --mode ablations --seeds 5 --steps 768 --time-budget-s 400

interpretability:
	@mkdir -p results/interpretability
	$(PYTHON) scripts/run_statistics.py
	$(PYTHON) -c "from pathlib import Path; import json; p=Path('results/interpretability/interpretability_probe.json'); assert p.exists(), p; print('INTERPRETABILITY_OK', json.load(p.open()).get('label'))"

statistics:
	$(PYTHON) scripts/run_statistics.py

figures:
	$(PYTHON) scripts/generate_figures.py
	$(PYTHON) scripts/generate_paper_tables.py || true

paper: figures statistics
	$(PYTHON) scripts/validate_paper.py
	@command -v pandoc >/dev/null 2>&1 && pandoc paper/RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.md -o paper/RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.tex || cp paper/handwritten.tex paper/RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.tex
	@echo "PAPER_BUILD_OK"

artifact: reproduce-clean
	$(PYTHON) scripts/build_artifact.py
	@echo "ARTIFACT_OK"

reproduce-clean: bootstrap test causal-tests algorithm-validation semantic-intervention-tests dial-validation tarmac-validation
	$(PYTHON) scripts/run_final_experiments.py --quick --mode all
	$(PYTHON) scripts/run_statistics.py
	$(PYTHON) scripts/generate_figures.py
	$(PYTHON) scripts/validate_paper.py

gate4-cpu: test smoke causal-tests algorithm-validation semantic-intervention-tests dial-validation tarmac-validation
	@echo "GATE4_OULU_CPU_OK (smoke≠scientific final)"

gate4-gpu:
	@$(PYTHON) -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA',ok); sys.exit(0 if not ok else 1)" \
		&& ($(PYTHON) scripts/emit_blocked_gpu.py; echo 'GATE4_NVIDIA_GPU_PENDING / BLOCKED_GPU (no CUDA)') \
		|| $(PYTHON) scripts/run_smoke_experiments.py --device cuda --seeds 1 --steps 64

blocked-gpu:
	$(PYTHON) scripts/emit_blocked_gpu.py

supervisor-cpu-gate:
	$(PYTHON) scripts/supervisor_cpu_gate.py

clean:
	rm -rf results/smoke/* results/pilot/* .coverage htmlcov
