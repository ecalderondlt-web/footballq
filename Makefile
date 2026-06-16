.PHONY: lint test synthetic-demo

lint:
	ruff check .

test:
	pytest

synthetic-demo:
	python scripts/run_synthetic_demo.py

