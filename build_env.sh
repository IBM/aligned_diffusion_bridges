#!/bin/bash

source $(conda info --root)/etc/profile.d/conda.sh

conda create -y --name sbalign python=3.9
conda activate sbalign

python -m pip install --upgrade pip
conda env update --file $PWD/env.yml

TORCH=1.12.0
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CUDA=cu113
else
    CUDA=cpu
fi

ARCH=$(uname -m)

python -m pip install torch_geometric

if [[ "$ARCH" == "arm64" ]]; then
    python -m pip install torch_scatter
    python -m pip install torch_sparse
    python -m pip install torch_cluster
else
    python -m pip install --no-index \
        torch_scatter \
        torch_sparse \
        torch_cluster \
        -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
fi

python -m pip install --upgrade e3nn
python setup.py develop
