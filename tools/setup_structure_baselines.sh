#!/usr/bin/env bash

# Create an isolated environment for the reviewer-requested structure baselines.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BASE_PYTHON=${BASE_PYTHON:-/path/to/conda/envs/lmpro-nano-torch26-cu124/bin/python}
VENV_DIR=${VENV_DIR:-$ROOT/../.venvs/structure-baselines}
PACKAGE_DIR=${PACKAGE_DIR:-$ROOT/../.packages/structure-baselines}

cd "$ROOT"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install \
    pytorch-lightning==2.5.2 \
    torchmetrics==1.3.2 \
    transformers==4.40.2

igfold_wheel="$PACKAGE_DIR/igfold-0.4.0-py3-none-any.whl"
antiberty_wheel="$PACKAGE_DIR/antiberty-0.1.3-py3-none-any.whl"
if [ ! -f "$igfold_wheel" ] || [ ! -f "$antiberty_wheel" ]; then
    mkdir -p "$PACKAGE_DIR"
    "$VENV_DIR/bin/python" -m pip download --no-deps --only-binary=:all: \
        --dest "$PACKAGE_DIR" antiberty==0.1.3 igfold==0.4.0
fi
echo "ddf73636cf86ca2ef91eab25f79bf8052a3deaeae0ee4e056c8eecd152582c9c  $igfold_wheel" | sha256sum -c -
echo "30d910992b190013871bac49cdc032e01a19339f7d2b958ab99b0eb44638352a  $antiberty_wheel" | sha256sum -c -
"$VENV_DIR/bin/python" -m pip install --no-deps "$antiberty_wheel" "$igfold_wheel"

echo "structure baseline environment: $VENV_DIR"
echo "IgFold uses the strict state-dict compatibility loader in tools/run_nanobody_structure_benchmark.py"
echo "OpenMM refinement is intentionally omitted; reviewer predictions use --refine=false"

# NanoBodyBuilder2 additions (2026-08-09). ImmuneBuilder is pinned to commit
# 0df4e2ad82a1aa60f37ea9dae335d1198159ef78 (PyPI-reported version 1.2); the
# cached tarball SHA-256 is 70545a467803da5b928a4fa09f0df17b6e690a9ef9ac07dc0e2f221426ef15d2.
# ANARCI needs an hmmscan binary; a project-local HMMER 3.4 prefix is used.
if [ "${WITH_NANOBODYBUILDER2:-0}" = 1 ]; then
    immunebuilder_tarball="$PACKAGE_DIR/ImmuneBuilder-0df4e2ad.tar.gz"
    if [ ! -f "$immunebuilder_tarball" ]; then
        mkdir -p "$PACKAGE_DIR"
        curl -sL -o "$immunebuilder_tarball" \
            "https://codeload.github.com/oxpig/ImmuneBuilder/tar.gz/0df4e2ad82a1aa60f37ea9dae335d1198159ef78"
    fi
    echo "70545a467803da5b928a4fa09f0df17b6e690a9ef9ac07dc0e2f221426ef15d2  $immunebuilder_tarball" | sha256sum -c -
    "$VENV_DIR/bin/python" -m pip install "$immunebuilder_tarball"
    "$VENV_DIR/bin/python" -m pip install anarci==2026.2.13.2 openmm==8.5.2 pdbfixer==1.12.0
    if [ ! -x "$ROOT/../.venvs/hmmer/bin/hmmscan" ]; then
        conda create -y -p "$ROOT/../.venvs/hmmer" -c bioconda hmmer
    fi
    echo "NanoBodyBuilder2 weights must be MD5-verified into baselines/structure_predictors/ImmuneBuilder_weights/"
    echo "(tools/download_immunebuilder_weights.sh where zenodo.org is reachable)"
fi
