ENV=.env
PYTHON=$(ENV)/bin/python
PIP=$(ENV)/bin/pip
SITE_PACKAGES=$(wildcard $(ENV)/lib/python*/site-packages)
REQUIREMENTS=requirements.txt
STREAMLIT=$(ENV)/bin/streamlit
BLACK=$(ENV)/bin/black
ISORT=$(ENV)/bin/isort
MYPY=$(ENV)/bin/mypy
PYTHON_FILES=$(shell find -type f -name '*.py' -not -path './.env/*' -not -path './CUB/*')
# codalab address for downloading the main CUB dataset
CODAENV=.codaenv
CODA_PIP=$(CODAENV)/bin/pip
CODALAB=$(CODAENV)/bin/cl
CUB_HASH=0xd013a7ba2e88481bbc07e787f73109f5

SSH_DESTINATION=root@ssh5.vast.ai
SSH_PORT=11820
SSH_DIRECTORY=$(USER)-hiddeninfo-sync

.PHONY: run
run: $(ENV) $(SITE_PACKAGES) $(STREAMLIT)
	$(STREAMLIT) run main.py

.PHONY: format
format: $(BLACK)
	$(BLACK) $(PYTHON_FILES)
	$(ISORT) -sl $(PYTHON_FILES)

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
	$(PIP) install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
	$(PIP) install torchvision scipy scikit-learn

# Sync local files with a remote location.
.PHONY: ssh-sync
ssh-sync:
	rsync -rv \
		--filter ':- .gitignore' \
		--exclude ".git" \
		-e 'ssh -p $(SSH_PORT)' \
		. $(SSH_DESTINATION):$(SSH_DIRECTORY)

# Run a make command on a remote location.
# E.g.: make COMMAND="run" ssh-run
.PHONY: ssh-run
ssh-run:
	ssh -p $(SSH_PORT) $(SSH_DESTINATION) \
		"cd $(SSH_DIRECTORY) && make $(COMMAND)"