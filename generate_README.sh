#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-05-29 23:23:44 (UTC+0200)

func runcmd() {
    OUTPUT=$(eval $1)
    echo "\`\`\`"
    echo "$ $1\n"
    echo "$OUTPUT"
    echo "\`\`\`"
}

[ -d figures ] && rm -r figures

cat << EOF
# Self-Organizing Map
PyTorch implementation of a Self-Organizing Map.
The implementation makes possible the use of a GPU if available for faster computations.
It follows the scikit package semantics for training and usage of the model.

EOF

cat << EOF
# Requirements
The SOM object requires numpy, scipy and torch installed.

The graph-based clustering requires scikit-learn and the image-based clustering requires scikit-image. By default,
we use the graph-based clustering

The toy example uses scikit-learn for the toy dataset generation

The MD application requires pymol for loading the trajectory

Then one can run :
\`\`\`
pip install quicksom
\`\`\`
EOF

cat << EOF
# SOM object interface
The SOM object can be created using any grid size, with a optional periodic topology.
One can also choose optimization parameters such as the number of epochs to train or the batch size
EOF

cat << EOF
\`\`\`
from quicksom import SOM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = torch.from_numpy(X)
X = X.to(device)
m, n = 50, 50
dim = X.shape[1]
n_iter = 5
batch_size = 10
som = SOM(m, n, dim, n_iter, device=device, periodic=True)
learning_error = som.fit(X, batch_size=batch_size)
bmus, inference_error = som.predict(X, batch_size=batch_size)
predicted_clusts, errors = som.predict_cluster(X)
\`\`\`
EOF

runcmd "./main.py"

cp -r figs figures

cat << EOF
## Input dataset:
![input](https://raw.githubusercontent.com/bougui505/quicksom/master/figures/moons.png)
## Umatrix:
![Umatrix](https://raw.githubusercontent.com/bougui505/quicksom/master/figures/umat.png)
## Data projection:
![project](https://raw.githubusercontent.com/bougui505/quicksom/master/figures/project.png)
## Cluster projection:
![project](https://raw.githubusercontent.com/bougui505/quicksom/master/figures/project_clusts.png)
## Cluster affectation:
![project](https://raw.githubusercontent.com/bougui505/quicksom/master/figures/clusts.png)
EOF
