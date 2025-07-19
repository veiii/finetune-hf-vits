# Finetune HF VITS Makefile
# This Makefile provides convenient commands for setting up, training, and testing VITS models

.PHONY: help install install-deps install-phonemizer install-uroman setup-cython clean
.PHONY: finetune finetune-local finetune-parquet test-model test-inference
.PHONY: convert-checkpoint convert-mms-checkpoint

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Installation:"
	@echo "  install           - Complete installation (deps + cython + optional tools)"
	@echo "  install-deps      - Install Python dependencies"
	@echo "  install-phonemizer- Install phonemizer (for VITS models)"
	@echo "  install-uroman     - Install uroman (for some MMS models)"
	@echo "  setup-cython      - Build Cython monotonic alignment search"
	@echo ""
	@echo "Model Conversion:"
	@echo "  convert-mms-checkpoint - Convert MMS checkpoint for training (requires LANG_CODE)"
	@echo "  convert-checkpoint     - Convert custom checkpoint (requires CHECKPOINT_PATH & GENERATOR_PATH)"
	@echo ""
	@echo "Training:"
	@echo "  finetune          - Run finetuning with default config"
	@echo "  finetune-local    - Run finetuning with local Polish dataset config"
	@echo "  finetune-parquet  - Run finetuning with parquet file dataset"
	@echo ""
	@echo "Testing:"
	@echo "  test-model        - Test trained model with vist_server on Pan Tadeusz"
	@echo "  test-inference    - Test inference with default MMS Polish model"
	@echo ""
	@echo "Utilities:"
	@echo "  clean             - Clean build artifacts and cache"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make convert-mms-checkpoint LANG_CODE=pol OUTPUT_DIR=./model_files/pol"
	@echo "  make finetune-local"
	@echo "  make test-model MODEL_PATH=/path/to/model"

# Installation targets
install: install-deps setup-cython
	@echo "✓ Complete installation finished!"

install-deps:
	@echo "Installing Python dependencies..."
	@echo "Installing system packages for Cython compilation..."
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y python3-dev build-essential; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum install -y python3-devel gcc gcc-c++ make; \
	elif command -v dnf >/dev/null 2>&1; then \
		sudo dnf install -y python3-devel gcc gcc-c++ make; \
	else \
		echo "⚠ Please install Python development headers manually:"; \
		echo "   Ubuntu/Debian: sudo apt-get install python3-dev build-essential"; \
		echo "   RHEL/CentOS: sudo yum install python3-devel gcc gcc-c++ make"; \
		echo "   Fedora: sudo dnf install python3-devel gcc gcc-c++ make"; \
	fi
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

install-phonemizer:
	@echo "Installing phonemizer dependencies..."
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y festival espeak-ng mbrola; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum install -y festival espeak-ng mbrola; \
	else \
		echo "⚠ Please install festival, espeak-ng, and mbrola manually for your system"; \
	fi
	pip install phonemizer
	@echo "✓ Phonemizer installed"

install-uroman:
	@echo "Installing uroman..."
	@if [ ! -d "uroman" ]; then \
		git clone https://github.com/isi-nlp/uroman.git; \
		export UROMAN=$$(pwd)/uroman; \
		echo "export UROMAN=$$(pwd)/uroman" >> ~/.bashrc; \
		echo "✓ Uroman installed. Please run 'source ~/.bashrc' or restart your shell"; \
	else \
		echo "✓ Uroman already exists"; \
	fi

setup-cython:
	@echo "Building Cython monotonic alignment search..."
	cd monotonic_align && \
	mkdir -p monotonic_align && \
	python setup.py build_ext --inplace
	@echo "✓ Cython setup complete"

# Model conversion targets
convert-mms-checkpoint:
	@if [ -z "$(LANG_CODE)" ]; then \
		echo "❌ Please specify LANG_CODE: make convert-mms-checkpoint LANG_CODE=pol"; \
		echo "Available language codes: https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html"; \
		exit 1; \
	fi
	@echo "Converting MMS checkpoint for language: $(LANG_CODE)"
	@mkdir -p $(or $(OUTPUT_DIR),./model_files/$(LANG_CODE))
	python convert_original_discriminator_checkpoint.py \
		--language_code $(LANG_CODE) \
		--pytorch_dump_folder_path $(or $(OUTPUT_DIR),./model_files/$(LANG_CODE)) \
		$(if $(PUSH_TO_HUB),--push_to_hub $(PUSH_TO_HUB),)
	@echo "✓ MMS checkpoint converted and saved to: $(or $(OUTPUT_DIR),./model_files/$(LANG_CODE))"

convert-checkpoint:
	@if [ -z "$(CHECKPOINT_PATH)" ] || [ -z "$(GENERATOR_PATH)" ]; then \
		echo "❌ Please specify both CHECKPOINT_PATH and GENERATOR_PATH:"; \
		echo "   make convert-checkpoint CHECKPOINT_PATH=/path/to/discriminator.pth GENERATOR_PATH=/path/to/generator OUTPUT_DIR=/path/to/output"; \
		exit 1; \
	fi
	@echo "Converting custom checkpoint..."
	@mkdir -p $(or $(OUTPUT_DIR),./model_files/converted)
	python convert_original_discriminator_checkpoint.py \
		--checkpoint_path $(CHECKPOINT_PATH) \
		--generator_checkpoint_path $(GENERATOR_PATH) \
		--pytorch_dump_folder_path $(or $(OUTPUT_DIR),./model_files/converted) \
		$(if $(PUSH_TO_HUB),--push_to_hub $(PUSH_TO_HUB),)
	@echo "✓ Custom checkpoint converted and saved to: $(or $(OUTPUT_DIR),./model_files/converted)"

# Training targets
finetune:
	@echo "Starting finetuning with default English config..."
	accelerate launch run_vits_finetuning.py ./training_config_examples/finetune_english.json

finetune-local:
	@echo "Starting finetuning with local dataset..."
	@CONFIG_FILE=$(or $(CONFIG_PATH),"./training_config_examples/finetune_mms_pol_local_dataset.json"); \
	if [ ! -f "$$CONFIG_FILE" ]; then \
		echo "❌ Config file not found: $$CONFIG_FILE"; \
		if [ -z "$(CONFIG_PATH)" ]; then \
			echo "💡 You can specify a custom config with: make finetune-local CONFIG_PATH=/path/to/config.json"; \
		fi; \
		exit 1; \
	fi
	@echo "📋 Validating dataset paths from config file..."
	@# Check if jq is available
	@if ! command -v jq >/dev/null 2>&1; then \
		echo "❌ jq is required but not installed. Please install jq:"; \
		echo "   Ubuntu/Debian: sudo apt-get install jq"; \
		echo "   macOS: brew install jq"; \
		echo "   Or download from: https://github.com/stedolan/jq/releases"; \
		exit 1; \
	fi
	@CONFIG_FILE=$(or $(CONFIG_PATH),"./training_config_examples/finetune_mms_pol_local_dataset.json"); \
	ERRORS=""; \
	\
	MODEL_PATH=$$(jq -r '.model_name_or_path // ""' "$$CONFIG_FILE"); \
	DATA_DIR=$$(jq -r '.data_dir // ""' "$$CONFIG_FILE"); \
	DATASET_PATH=$$(jq -r '.dataset_path // ""' "$$CONFIG_FILE"); \
	PARQUET_FILE=$$(jq -r '.parquet_file // ""' "$$CONFIG_FILE"); \
	\
	if [ -n "$$MODEL_PATH" ] && [ "$$MODEL_PATH" != "null" ] && [ "$$MODEL_PATH" != "" ]; then \
		MODEL_EXPANDED=$$(echo "$$MODEL_PATH" | sed "s|^~|$$HOME|"); \
		if [ ! -e "$$MODEL_EXPANDED" ]; then \
			ERRORS="$$ERRORS\n   - Model path does not exist: $$MODEL_EXPANDED"; \
		fi; \
	fi; \
	\
	if [ -n "$$DATA_DIR" ] && [ "$$DATA_DIR" != "null" ] && [ "$$DATA_DIR" != "" ]; then \
		DATA_EXPANDED=$$(echo "$$DATA_DIR" | sed "s|^~|$$HOME|"); \
		if [ ! -d "$$DATA_EXPANDED" ]; then \
			ERRORS="$$ERRORS\n   - Data directory does not exist: $$DATA_EXPANDED"; \
		fi; \
	fi; \
	\
	if [ -n "$$DATASET_PATH" ] && [ "$$DATASET_PATH" != "null" ] && [ "$$DATASET_PATH" != "" ]; then \
		DATASET_EXPANDED=$$(echo "$$DATASET_PATH" | sed "s|^~|$$HOME|"); \
		if [ ! -d "$$DATASET_EXPANDED" ]; then \
			ERRORS="$$ERRORS\n   - Dataset path does not exist: $$DATASET_EXPANDED"; \
		fi; \
	fi; \
	\
	if [ -n "$$PARQUET_FILE" ] && [ "$$PARQUET_FILE" != "null" ] && [ "$$PARQUET_FILE" != "" ]; then \
		PARQUET_EXPANDED=$$(echo "$$PARQUET_FILE" | sed "s|^~|$$HOME|"); \
		if [ ! -f "$$PARQUET_EXPANDED" ]; then \
			ERRORS="$$ERRORS\n   - Parquet file does not exist: $$PARQUET_EXPANDED"; \
		fi; \
	fi; \
	\
	DATASET_SOURCES_COUNT=0; \
	if [ -n "$$DATA_DIR" ] && [ "$$DATA_DIR" != "null" ] && [ "$$DATA_DIR" != "" ]; then \
		DATASET_SOURCES_COUNT=$$((DATASET_SOURCES_COUNT + 1)); \
	fi; \
	if [ -n "$$DATASET_PATH" ] && [ "$$DATASET_PATH" != "null" ] && [ "$$DATASET_PATH" != "" ]; then \
		DATASET_SOURCES_COUNT=$$((DATASET_SOURCES_COUNT + 1)); \
	fi; \
	if [ -n "$$PARQUET_FILE" ] && [ "$$PARQUET_FILE" != "null" ] && [ "$$PARQUET_FILE" != "" ]; then \
		DATASET_SOURCES_COUNT=$$((DATASET_SOURCES_COUNT + 1)); \
	fi; \
	\
	if [ $$DATASET_SOURCES_COUNT -eq 0 ]; then \
		ERRORS="$$ERRORS\n   - No valid dataset source specified (data_dir, dataset_path, or parquet_file)"; \
	fi; \
	\
	if [ -n "$$ERRORS" ]; then \
		echo "❌ Dataset validation failed:"; \
		echo -e "$$ERRORS"; \
		exit 1; \
	else \
		echo "✓ All dataset paths validated successfully"; \
	fi
	@CONFIG_FILE=$(or $(CONFIG_PATH),"./training_config_examples/finetune_mms_pol_local_dataset.json"); \
	echo "🚀 Starting finetuning with config: $$CONFIG_FILE"; \
	accelerate launch run_vits_finetuning.py "$$CONFIG_FILE"

finetune-parquet:
	@echo "Starting finetuning with parquet dataset..."
	@if [ -z "$(PARQUET_FILE)" ]; then \
		echo "❌ Please specify PARQUET_FILE: make finetune-parquet PARQUET_FILE=/path/to/dataset.parquet"; \
		exit 1; \
	fi
	accelerate launch run_vits_finetuning.py \
		--model_name_or_path $(or $(MODEL_PATH),facebook/mms-tts-pol) \
		--parquet_file $(PARQUET_FILE) \
		--output_dir $(or $(OUTPUT_DIR),./output) \
		--do_train true \
		--num_train_epochs $(or $(EPOCHS),50) \
		--per_device_train_batch_size $(or $(BATCH_SIZE),4) \
		--learning_rate $(or $(LR),2e-5)

# Testing targets
test-model:
	@echo "Testing trained model with Pan Tadeusz..."
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "❌ Please specify MODEL_PATH: make test-model MODEL_PATH=/path/to/model"; \
		exit 1; \
	fi
	@if [ ! -f "testing_dataset/pan_tadeusz.txt" ]; then \
		echo "❌ Test file not found: testing_dataset/pan_tadeusz.txt"; \
		exit 1; \
	fi
	python utils/tts/vist_server.py \
		--input testing_dataset/pan_tadeusz.txt \
		--output $(or $(OUTPUT_AUDIO),./test_output_$(shell date +%Y%m%d_%H%M%S).mp3) \
		--model $(MODEL_PATH) \
		--max-lines $(or $(MAX_LINES),50)
	@echo "✓ Test completed! Audio saved to: $(or $(OUTPUT_AUDIO),./test_output_*.mp3)"

test-inference:
	@echo "Testing inference with default MMS Polish model..."
	@if [ ! -f "testing_dataset/pan_tadeusz.txt" ]; then \
		echo "❌ Test file not found: testing_dataset/pan_tadeusz.txt"; \
		exit 1; \
	fi
	python utils/tts/vist_server.py \
		--input testing_dataset/pan_tadeusz.txt \
		--output $(or $(OUTPUT_AUDIO),./test_default_$(shell date +%Y%m%d_%H%M%S).mp3) \
		--model facebook/mms-tts-pol \
		--max-lines $(or $(MAX_LINES),10)
	@echo "✓ Default model test completed!"

# Utility targets
clean:
	@echo "Cleaning build artifacts and cache..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.so" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	@echo "✓ Cleanup complete"

# Advanced targets with parameters
finetune-custom:
	@echo "Starting custom finetuning..."
	@if [ -z "$(CONFIG)" ]; then \
		echo "❌ Please specify CONFIG: make finetune-custom CONFIG=/path/to/config.json"; \
		exit 1; \
	fi
	accelerate launch run_vits_finetuning.py $(CONFIG)

# Quick setup for new users
quickstart: install
	@echo ""
	@echo "🚀 Quick setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Convert MMS checkpoint: make convert-mms-checkpoint LANG_CODE=pol"
	@echo "2. For Polish finetuning: make finetune-local"
	@echo "3. For testing: make test-inference"
	@echo "4. For custom dataset: make finetune-parquet PARQUET_FILE=/path/to/your/dataset.parquet"
	@echo ""
