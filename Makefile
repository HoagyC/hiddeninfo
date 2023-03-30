ENV=.env
PYTHON=$(ENV)/bin/python
PIP=$(ENV)/bin/pip
SITE_PACKAGES=$(wildcard $(ENV)/lib/python*/site-packages)
REQUIREMENTS=requirements.txt
STREAMLIT=$(ENV)/bin/streamlit
BLACK=$(ENV)/bin/black
ISORT=$(ENV)/bin/isort
MYPY=$(ENV)/bin/mypy
PYTHON_FILES=$(shell find -type f -name '*.py' -not -path './.env/*')
# codalab address for downloading the main CUB dataset
CODAENV=.codaenv
CODA_PIP=$(CODAENV)/bin/pip
CODALAB=$(CODAENV)/bin/cl
CUB_HASH=0xd013a7ba2e88481bbc07e787f73109f5

SSH_PORT=12940
VASTAI_N=4
SSH_DESTINATION=root@ssh$(VASTAI_N).vast.ai
SSH_DIRECTORY=hoagy-hiddeninfo-sync

# List of ports for different vast.ai servers.
# SSH_PORTS = 11082
# VASTAI_NS = 6

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

.PHONY: hoagy
hoagy:
	git config --global user.email "hoagycunningham@gmail.com"
	git config --global user.name "hoagyc"

# TODO: Replace this with proper make dependencies.
.PHONY: install
install:
	apt install -y python3.8-venv
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
	$(PYTHON) CUB/generate_new_data.py

# Sync local files with a remote location.
# Run with ssh-sync PORT=.
.PHONY: ssh-sync
ssh-sync:
	rsync -rv \
		--filter ':- .gitignore' \
		--exclude ".git" \
		-e 'ssh -p $(SSH_PORT)' \
		. $(SSH_DESTINATION):$(SSH_DIRECTORY)
	
.PHONY: secret
secret:
	scp -P $(SSH_PORT) secrets.json $(SSH_DESTINATION):/root/$(SSH_DIRECTORY)

# Run a make command on a remote location.
# E.g.: make COMMAND="run" ssh-run
.PHONY: ssh-run
ssh-run:
	ssh -p $(SSH_PORT) $(SSH_DESTINATION) \
		"cd $(SSH_DIRECTORY) && pwd && make $(COMMAND)"


.PHONY: aws-pull
aws-pull:
	apt install unzip
	$(PYTHON) CUB/aws_download.py
	unzip -n CUB_dataset.zip # -n to not overwrite existing files

# Get a file through scp from the remote location.
.PHONY: retrieve
retrieve:
	scp -P $(SSH_PORT) $(SSH_DESTINATION):/root/$(SSH_DIRECTORY)/$(FILE) .

.PHONY: ssh-setup
ssh-setup:
	ssh -oStrictHostKeyChecking=no -p $(SSH_PORT) $(SSH_DESTINATION) \
		"apt install -y build-essential python3-venv"
	make ssh-sync
	make ssh-run COMMAND="install"
	make secret
	make ssh-run COMMAND="aws-pull"
