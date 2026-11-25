PYTHON ?= .venv/bin/python
PYTHON_BOOTSTRAP ?= python3.12
UV ?= uv
GPUS ?= 0
GPU_HOURS ?= 96
RUN = $(PYTHON) -m quinn.run
GPU_FLAGS = --gpus $(GPUS) --gpu-hour-budget $(GPU_HOURS) --resume

.PHONY: setup setup-gpu fetch-upstream corpus test hardware-check smoke-test pilot characterize evaluate-measurement evaluate-scheduler evaluate-system ablation generalization train-rl figures tables report reproduce

setup:
	$(UV) venv --allow-existing --python $(PYTHON_BOOTSTRAP) .venv
	$(UV) -q pip install --python $(PYTHON) -r requirements-cpu.txt
	$(UV) -q pip install --python $(PYTHON) --no-deps -e .

setup-gpu:
	$(UV) -q pip install --python $(PYTHON) '.[gpu]'

fetch-upstream:
	./scripts/fetch_upstream.sh

corpus: fetch-upstream
	$(RUN) --corpus .upstream/kernelbench

test:
	$(PYTHON) -m pytest -q

hardware-check:
	$(RUN) --hardware-check

smoke-test:
	$(RUN) --config configs/smoke.yaml $(GPU_FLAGS)

pilot:
	$(RUN) --config configs/pilot.yaml $(GPU_FLAGS)

characterize:
	$(RUN) --config configs/characterization.yaml $(GPU_FLAGS)

evaluate-measurement:
	@test -n "$(CAPTURES)" || (echo 'Set CAPTURES to raw A100 capture directories including independent gold sessions.' && exit 2)
	$(RUN) --analyze-capture $(CAPTURES)

evaluate-scheduler:
	$(RUN) --config configs/scheduler.yaml --resume
	$(RUN) --report

evaluate-system:
	$(RUN) --config configs/evaluation.yaml --session service-capture-$(subst $(comma),-,$(GPUS)) $(GPU_FLAGS)

ablation: evaluate-scheduler

generalization:
	$(RUN) --config configs/evaluation.yaml $(GPU_FLAGS)

train-rl:
	$(RUN) --config configs/rl.yaml $(GPU_FLAGS)

figures tables report:
	$(RUN) --report

reproduce: test hardware-check corpus
	$(RUN) --config configs/scheduler.yaml --recompute
	$(RUN) --report

comma := ,
