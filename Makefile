PYTHON ?= python

.PHONY: install test run-all lint

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

run-all:
	$(PYTHON) -m src.cli all --config configs/config.yaml

lint:
	flake8 src tests
