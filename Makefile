# CatBowlWatch — dataset & build orchestration.
#
# Stages (Phase 1):
#   make collect    sample frames from videos in $(RAW_INCOMING) → $(RAW_LABELLED)/images
#   make organise   separate intermixed jpg/txt files in $(RAW_LABELLED)
#   make validate   sanity-check label files
#   make split      random seeded train/val/test split + data/data.yaml
#   make data       organise → validate → split
#   make clean-data wipe split outputs (raw stays intact)
#   make test       run pytest
#
# Override variables on the command line, e.g.:
#   make split SPLIT_RATIOS="0.8 0.1 0.1" SEED=7
#   make collect INTERVAL=0.5

PYTHON       ?= poetry run python
RAW_INCOMING ?= data/raw/incoming
RAW_LABELLED ?= data/raw/labelled
DATA_ROOT    ?= data
SPLIT_RATIOS ?= 0.70 0.15 0.15
SEED         ?= 42
INTERVAL     ?= 1.0

VIDEO_EXTS := mp4 mov m4v MP4 MOV M4V

.PHONY: help collect organise validate split data clean-data train export-onnx test

help:  ## Show available targets.
	@awk -F':.*?## ' '/^[a-zA-Z_-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

collect:  ## Sample frames from videos in $(RAW_INCOMING) into $(RAW_LABELLED)/images.
	@mkdir -p $(RAW_LABELLED)/images
	@found=0; \
	for ext in $(VIDEO_EXTS); do \
	  for f in $(RAW_INCOMING)/*.$$ext; do \
	    [ -f "$$f" ] || continue; \
	    found=1; \
	    echo ">> sampling $$f"; \
	    $(PYTHON) scripts/collect_data.py --source "$$f" --out $(RAW_LABELLED)/images --interval $(INTERVAL); \
	  done; \
	done; \
	if [ $$found -eq 0 ]; then echo "No videos found under $(RAW_INCOMING)/ (looked for: $(VIDEO_EXTS))"; fi

organise:  ## Separate intermixed jpg/txt files in $(RAW_LABELLED) into images/ + labels/.
	$(PYTHON) scripts/organise_raw.py --src $(RAW_LABELLED) --dst $(RAW_LABELLED)

validate:  ## Sanity-check label files (class IDs, coord ranges, image-label pairing).
	$(PYTHON) scripts/validate_labels.py --images $(RAW_LABELLED)/images --labels $(RAW_LABELLED)/labels

split:  ## Random $(SPLIT_RATIOS) split into $(DATA_ROOT)/{images,labels}/{train,val,test}/ + data.yaml.
	$(PYTHON) scripts/split_dataset.py --src $(RAW_LABELLED) --dst $(DATA_ROOT) --ratios $(SPLIT_RATIOS) --seed $(SEED)

data: organise validate split  ## Full pipeline: organise → validate → split.

clean-data:  ## Remove split outputs and data.yaml. Raw data is untouched.
	rm -rf $(DATA_ROOT)/images/train $(DATA_ROOT)/images/val $(DATA_ROOT)/images/test
	rm -rf $(DATA_ROOT)/labels/train $(DATA_ROOT)/labels/val $(DATA_ROOT)/labels/test
	rm -f  $(DATA_ROOT)/data.yaml

train:  ## Train YOLOv8n on the dataset. Requires `poetry install --with training`.
	$(PYTHON) training/train.py

export-onnx:  ## Export trained .pt → ONNX opset 17 (verifies shape). Requires `poetry install --with training`.
	$(PYTHON) training/export.py

test:  ## Run unit tests.
	$(PYTHON) -m pytest tests/ -v
