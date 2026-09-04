# One-command(ish) entry points for the "Reproducing this" steps in README.md.
# Every target here is a thin wrapper around the exact commands documented in
# README.md / eval/README.md — read those if you want to run steps by hand,
# or to understand what each target actually does before running it.
#
#   make help        list targets
#   make setup       clone upstream Ternlight + create venv + install everything
#   make eval-quick  validate the harness in ~10 min, no training required
#   make train       run both training phases (~50-60 min on Apple Silicon)
#   make eval        run the full 3-way eval against your trained checkpoint

SHELL := /bin/bash
PYTHON3 := python3
UPSTREAM_DIR := upstream
DISTILL_DIR := $(UPSTREAM_DIR)/training/distill
VENV := venv
VENV_PY := $(VENV)/bin/python3
VENV_PIP := $(VENV)/bin/pip

# Auto-discovers the QAT checkpoint from a completed `make train-qat` run so
# `make eval` works with no arguments. Override explicitly if you trained more
# than one QAT run: `make eval CKPT=upstream/training/distill/runs/.../checkpoint_ep30.pt`
CKPT ?= $(firstword $(wildcard $(DISTILL_DIR)/runs/techdocs-qat-*/checkpoint_ep30.pt))

.PHONY: help setup clone-upstream venv install bridge-install prepare \
        train-fp32 train-qat train eval-quick eval clean

help:
	@echo "Targets:"
	@echo "  setup          clone upstream Ternlight, create venv, install Python + Node deps"
	@echo "  clone-upstream clone https://github.com/soycaporal/ternlight into ./upstream"
	@echo "  install        pip install both requirements files into ./venv"
	@echo "  bridge-install npm ci in ternlight-bridge/ (deterministic, uses package-lock.json)"
	@echo "  prepare        copy configs/ into upstream and run prep/prepare.py"
	@echo "  train-fp32     train the fp32 baseline (~25 min on Apple Silicon MPS)"
	@echo "  train-qat      train the ternary QAT model (~25 min on Apple Silicon MPS)"
	@echo "  train          train-fp32 + train-qat"
	@echo "  eval-quick     run eval with --skip-ours (~10 min, no checkpoint needed)"
	@echo "  eval           run the full 3-way eval (needs CKPT= or a completed train-qat)"
	@echo "  clean          remove venv/, upstream/, node_modules/ (all .gitignore'd, regenerable)"
	@echo ""
	@echo "See README.md and eval/README.md for the manual, numbered version of each step."

setup: install bridge-install
	@echo "Setup complete. Try 'make eval-quick' next (no training required),"
	@echo "or 'make train' to reproduce the full results table."

clone-upstream:
	@test -d $(UPSTREAM_DIR) || git clone https://github.com/soycaporal/ternlight $(UPSTREAM_DIR)

$(VENV)/bin/activate:
	$(PYTHON3) -m venv $(VENV)

venv: $(VENV)/bin/activate

install: venv clone-upstream
	$(VENV_PIP) install -r $(DISTILL_DIR)/requirements.txt
	$(VENV_PIP) install -r requirements.txt

bridge-install:
	cd ternlight-bridge && npm ci

prepare: install
	cp configs/techdocs*.yaml $(DISTILL_DIR)/configs/
	cd $(DISTILL_DIR) && $(CURDIR)/$(VENV_PY) prep/prepare.py --config configs/techdocs.yaml

train-fp32: prepare
	cd $(DISTILL_DIR) && $(CURDIR)/$(VENV_PY) train.py --config configs/techdocs-fp32.yaml

train-qat: prepare
	cd $(DISTILL_DIR) && $(CURDIR)/$(VENV_PY) train.py --config configs/techdocs-qat.yaml

train: train-fp32 train-qat

eval-quick: bridge-install
	cd eval && $(CURDIR)/$(VENV_PY) eval_retrieval.py --skip-ours

eval: bridge-install
ifeq ($(strip $(CKPT)),)
	$(error No checkpoint found under $(DISTILL_DIR)/runs/techdocs-qat-*/checkpoint_ep30.pt. \
	Run 'make train-qat' first, or pass CKPT=/path/to/checkpoint_ep30.pt)
endif
	cd eval && $(CURDIR)/$(VENV_PY) eval_retrieval.py --ckpt $(CURDIR)/$(CKPT)

clean:
	rm -rf $(VENV) $(UPSTREAM_DIR) ternlight-bridge/node_modules
