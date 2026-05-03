# Reproducible MLOps targets. Every published number in the report is
# traceable to one of these targets. See REPRODUCING.md for the canonical
# end-to-end recipe.

# ── Paths (override on the command line, e.g. `make benchmark S1_DIR=...`) ──
S1_DIR     ?= app/sample_data/S1
LABEL_DIR  ?= app/sample_data/Labels
SPLITS_DIR ?= app/sample_data/splits
SEGFORMER_CKPT ?= checkpoints/segformer_flood_best.pt
SAMPLE_SCENE   ?= $(S1_DIR)/Mekong_333434_S1Hand.tif

CALIBRATION_JSON := mlops/calibration.json
RESULTS_DIR      := mlops/results
FIG_DIR          := mlops/figures

PYTHON ?= python

.DEFAULT_GOAL := help


# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "Targets (run in this order for full reproduction):"
	@echo ""
	@echo "  setup            Install pinned Python dependencies"
	@echo "  download-data    Pull the full Sen1Floods11 hand-labeled split (~2 GB)"
	@echo "  download-model   Pull the trained SegFormer checkpoint from Hugging Face"
	@echo "  train-segformer  Train SegFormer from scratch (ClearML-tracked, ~1 hr on L4)"
	@echo "  calibrate        Compute the empirical ambiguity band (Figure A)"
	@echo "  benchmark        Run cascade efficiency benchmark (Figure B + system table)"
	@echo "  pipeline         Run cascaded inference on a single scene"
	@echo "  demo             End-to-end live demo: calibrate + benchmark + pipeline"
	@echo ""
	@echo "  docker-build     Build the reproducible Docker image"
	@echo "  docker-demo      Run 'make demo' inside the Docker image"
	@echo ""
	@echo "  clean            Remove generated results, figures, calibration"
	@echo ""
	@echo "Variables (override with VAR=value):"
	@echo "  S1_DIR=$(S1_DIR)"
	@echo "  LABEL_DIR=$(LABEL_DIR)"
	@echo "  SPLITS_DIR=$(SPLITS_DIR)"
	@echo "  SEGFORMER_CKPT=$(SEGFORMER_CKPT)"


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r mlops/requirements.txt
	@echo ""
	@echo "Setup complete. Configure ClearML next:"
	@echo "    clearml-init"


.PHONY: download-data
download-data:
	@command -v gsutil >/dev/null 2>&1 || { \
		echo "gsutil not found. Install: pip install gsutil, or use the gcloud SDK."; exit 1; }
	mkdir -p data/sen1floods11/S1 data/sen1floods11/Labels data/sen1floods11/splits
	gsutil -m -q rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand     data/sen1floods11/S1
	gsutil -m -q rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand  data/sen1floods11/Labels
	gsutil -q cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_train_data.csv   data/sen1floods11/splits/
	gsutil -q cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_valid_data.csv   data/sen1floods11/splits/
	gsutil -q cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv    data/sen1floods11/splits/
	gsutil -q cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_bolivia_data.csv data/sen1floods11/splits/
	@echo "Sen1Floods11 hand-labeled split downloaded → data/sen1floods11/"


.PHONY: download-model
download-model:
	mkdir -p checkpoints
	$(PYTHON) -c "from huggingface_hub import hf_hub_download; \
	  p = hf_hub_download(repo_id='hardik56711/segformer_flood_detection', \
	                       filename='segformer_flood_best.pt', \
	                       local_dir='checkpoints'); print('saved to', p)"


# ─────────────────────────────────────────────────────────────────────────────
# Pipelines
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: train-segformer
train-segformer:
	$(PYTHON) -m mlops.train_segformer \
		--s1-dir     $(S1_DIR) \
		--label-dir  $(LABEL_DIR) \
		--splits-dir $(SPLITS_DIR) \
		--ckpt-dir   $(dir $(SEGFORMER_CKPT))


.PHONY: calibrate
calibrate:
	$(PYTHON) mlops/calibrate_ambiguity_band.py \
		--s1-dir    $(S1_DIR) \
		--label-dir $(LABEL_DIR) \
		--split-csv $(SPLITS_DIR)/flood_test_data.csv \
		--out-json  $(CALIBRATION_JSON) \
		--out-figure $(FIG_DIR)/ambiguity_calibration.png


BIMOD_TAUS    ?= 1.0,2.0,4.0,8.0,16.0
ALIGNMENT_DB  ?= 2.0

.PHONY: benchmark
benchmark:
	$(PYTHON) mlops/benchmark_cascade.py \
		--s1-dir          $(S1_DIR) \
		--label-dir       $(LABEL_DIR) \
		--splits-dir      $(SPLITS_DIR) \
		--segformer-ckpt  $(SEGFORMER_CKPT) \
		--calibration-json $(CALIBRATION_JSON) \
		--bimodality-thresholds $(BIMOD_TAUS) \
		--alignment-db   $(ALIGNMENT_DB) \
		--out-dir $(RESULTS_DIR) \
		--fig-dir $(FIG_DIR)


.PHONY: pipeline
pipeline:
	@if [ -z "$$MODEL_TASK_ID" ]; then \
		echo "Set MODEL_TASK_ID=<clearml-task-id-of-segformer-training>"; \
		echo "Or run 'make benchmark' first which uses local checkpoints directly."; \
		exit 1; \
	fi
	$(PYTHON) mlops/cascaded_inference_pipeline.py \
		--scene $(SAMPLE_SCENE) \
		--model-task-id $$MODEL_TASK_ID \
		--calibration-task-id $$CALIBRATION_TASK_ID \
		--out /tmp/scene_flood_mask.tif \
		--local


.PHONY: demo
demo: calibrate benchmark
	@echo ""
	@echo "════════════════════════════════════════════════════════════════════"
	@echo "  Demo complete. Inspect:"
	@echo "    - $(FIG_DIR)/ambiguity_calibration.png   (Figure A)"
	@echo "    - $(FIG_DIR)/figure_b_tradeoff.png       (Figure B)"
	@echo "    - $(RESULTS_DIR)/system_comparison.md    (headline table)"
	@echo "    - $(RESULTS_DIR)/benchmark.csv           (raw numbers)"
	@echo "  Plus all of the above are mirrored in the ClearML web UI."
	@echo "════════════════════════════════════════════════════════════════════"


# ─────────────────────────────────────────────────────────────────────────────
# Docker
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build:
	docker build -t flood-mlops:latest .


.PHONY: docker-demo
docker-demo:
	docker compose run --rm mlops make demo


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf $(RESULTS_DIR) $(FIG_DIR) $(CALIBRATION_JSON)
	@echo "Cleaned generated outputs."
