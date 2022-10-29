ENV=.env
PYTHON=$(ENV)/bin/python
PIP=$(ENV)/bin/pip
SITE_PACKAGES=$(wildcard $(ENV)/lib/python*/site-packages)
REQUIREMENTS=requirements.txt
STREAMLIT=$(ENV)/bin/streamlit
BLACK=$(ENV)/bin/black
MYPY=$(ENV)/bin/mypy
PYTHON_FILES=main.py experiments.py pages/adversarial.py

.PHONY: run
run: $(ENV) $(SITE_PACKAGES) $(STREAMLIT)
	$(STREAMLIT) run main.py

.PHONY: format
format: $(BLACK)
	$(BLACK) $(PYTHON_FILES)

.PHONY: check
check: $(MYPY)
	$(MYPY) $(PYTHON_FILES)

.PHONY: clean
clean:
	rm -r $(ENV)

# TODO: Replace this with proper make dependencies.
.PHONY: install
install:
	apt install python3.8-venv
	python3.8 -m venv $(ENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

.PHONY: cub
cub:
	# codalab doesn't work with >python3.6 due to pypi dataclasses issue
	python3.6 -m venv $(CODAENV)
	$(CODA_PIP) install --upgrade pip
	$(CODA_PIP) install codalab
	$(CODALAB) download $(CUB_HASH)
	$(PYTHON) data_processing.py

	$(PIP) install --upgrade pip
	$(PIP) install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
	$(PIP) install torchvision scipy