install:
	pip install -r requirements.txt

install-dev: install
	pip install -r dev.requirements.txt
	pre-commit install

lint:
	python -m pylint src
	python -m flake8 src

install-streamlit:
	pip install streamlit==1.2.0 st-annotated-text==2.0.0 --quiet

demo: install-streamlit
	streamlit run demonstrator/__main__.py
