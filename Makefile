ENV=.env
PYTHON=$(ENV)/bin/python
PIP=$(ENV)/bin/pip
SITE_PACKAGES=$(wildcard $(ENV)/lib/python*/site-packages)
REQUIREMENTS=requirements.txt
STREAMLIT=$(ENV)/bin/streamlit
BLACK=$(ENV)/bin/black
PYTHON_FILES=main.py

.PHONY: run
run: $(ENV) $(SITE_PACKAGES) $(STREAMLIT)
	$(STREAMLIT) run main.py

.PHONY: format
format: $(BLACK)
	$(BLACK) $(PYTHON_FILES)

.PHONY: clean
clean:
	rm -r $(ENV)

# TODO: Replace this with proper make dependencies.
.PHONY: install
install:
	python -m venv $(ENV)
	$(PIP) install -r $(REQUIREMENTS)