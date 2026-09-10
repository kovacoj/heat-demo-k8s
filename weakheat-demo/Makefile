# weakheat-demo -- all compute runs in Docker containers.
#
#   make dataset    export FE operators + generate Firedrake dataset
#   make train      train the weak-form neural surrogate
#   make train-data-only   supervised-only baseline
#   make evaluate   test metrics + presentation figures
#   make test       unit tests
#   make sanity     single Firedrake solve with checks

FIREDRAKE_IMAGE ?= cerit.io/kovacoj1/heat-firedrake:20260909-081208
ML_IMAGE ?= weakheat-ml:latest

# The upstream image ships PYTHONPATH="/opt/python-packages:" whose trailing
# colon puts the CWD on sys.path; a local directory named firedrake/ then
# shadows the installed package.  Always override it and keep cwd in /tmp.
FD_RUN = docker run --rm \
	-e PYTHONPATH=/opt/python-packages \
	-e PYTHONUNBUFFERED=1 \
	-e HOME=/tmp \
	-w /tmp \
	-v $(PWD):/work \
	$(FIREDRAKE_IMAGE)

# torch container runs as the host user so written files stay owned by us
ML_RUN = docker run --rm \
	-e PYTHONUNBUFFERED=1 \
	-u $$(id -u):$$(id -g) \
	-v $(PWD):/work \
	-w /work \
	$(ML_IMAGE)

.PHONY: dataset sanity train train-data-only evaluate test build-ml clean

build-ml:
	docker build -t $(ML_IMAGE) ml/

sanity:
	$(FD_RUN) python3 /work/firedrake/solve_case.py

dataset:
	$(FD_RUN) python3 /work/firedrake/export_operators.py
	$(FD_RUN) python3 /work/firedrake/generate_dataset.py

train:
	$(ML_RUN) python3 ml/train.py

train-data-only:
	$(ML_RUN) python3 ml/train.py --data-only --out checkpoints/data_only.pt

evaluate:
	$(ML_RUN) python3 ml/evaluate.py

benchmark:
	$(ML_RUN) python3 ml/benchmark.py

test:
	$(ML_RUN) python3 -m pytest tests -q

clean:
	rm -rf data/raw data/processed checkpoints/*.pt presentation
