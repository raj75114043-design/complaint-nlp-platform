.PHONY: install data lint test eval train api console up down

install:   ## Install dev dependencies
	pip install -r requirements-dev.txt
data:      ## Regenerate the synthetic complaint stream
	python -m cnlp.cli data
lint:
	ruff check .
test:
	pytest -q
eval:      ## Full evaluation with quality gate
	python -m cnlp.eval.run --gate
train:     ## Train a challenger; promote it only if it beats the champion
	python -m cnlp.cli train --promote-if-better
api:
	uvicorn cnlp.api:app --reload
console:
	streamlit run streamlit_app.py
up:
	cp -n .env.example .env || true; docker compose up --build -d
down:
	docker compose down
