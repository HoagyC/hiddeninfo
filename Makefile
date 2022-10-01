ENV=.env
PYTHON=$(ENV)/bin/python
PIP=$(ENV)/bin/pip
SITE_PACKAGES=$(wildcard $(ENV)/lib/python*/site-packages)
REQUIREMENTS=requirements.txt
BLACK=$(ENV)/bin/black
PYTHON_FILES=$(shell find hiddeninfo -name '*.py')

.PHONY: run
run: $(PYTHON)
	$(PYTHON) -m hiddeninfo

.PHONY: format
format: $(BLACK)
	$(BLACK) $(PYTHON_FILES)

.PHONY: clean
clean:
	rm -r $(ENV)

$(PYTHON): $(ENV) $(SITE_PACKAGES)
$(ENV):
	python -m venv $(ENV)
$(SITE_PACKAGES): $(REQUIREMENTS)
	$(PIP) install -r $(REQUIREMENTS) > /dev/null
$(BLACK): $(SITE_PACKAGES)