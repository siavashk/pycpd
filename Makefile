autoformat:
	# set -e
	# isort .
	# black --config pyproject.toml .

lint:
	# set -e
	# isort -c .
	# black --check --config pyproject.toml .

test:
	set -e
	pytest

dev:
	pip install pytest black isort twine wheel pdoc3

requirements:
	python -m pip install -r requirements.txt

build:
	pip install .

visualize:
	pip install matplotlib

doc:
	pdoc --output-dir docs/ --html --force pycpd 
	mv docs/pycpd/* docs/
	rm -rf docs/pycpd