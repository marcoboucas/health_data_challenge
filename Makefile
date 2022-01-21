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


run:
	python -m src run --dataset=val --size=-1 --ner_name=medcat --ner_path=./weights/ner_medcat.pkl --assessor_name=bert
