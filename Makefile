ENV=.env
PYTHON=$(ENV)/bin/python
PIP=$(ENV)/bin/pip
CODALAB=$(ENV)/bin/cl
SITE_PACKAGES=$(wildcard $(ENV)/lib/python*/site-packages)
REQUIREMENTS=requirements.txt
STREAMLIT=$(ENV)/bin/streamlit
BLACK=$(ENV)/bin/black
MYPY=$(ENV)/bin/mypy
PYTHON_FILES=main.py experiments.py
# codalab address for downloading the main CUB dataset
CUB_HASH=0xd013a7ba2e88481bbc07e787f73109f5

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
install:.
	python -m venv $(ENV)
	$(PIP) install -r $(REQUIREMENTS)

.PHONY: cub
cub:
	$(PIP) install --upgrade pip
	$(PIP) install codalab
	$(CODALAB) download $(CUB_HASH)
	$(PYTHON) CUB_200_2011/data_processing.py
	$(PIP) install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
	$(PIP) install torchvision