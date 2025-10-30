PYTHON ?= python

.PHONY: install test run-all lint prepare-dirs

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

prepare-dirs:
	mkdir -p reports/tables reports/figures models data/processed

run-all: prepare-dirs
	$(PYTHON) -m src.cli --config configs/config.yaml all

test:
	$(PYTHON) -m pytest -q

lint:
	flake8 src tests
