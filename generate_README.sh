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
#### Requirements and setup
The SOM object requires torch installed.

It has dependencies in numpy, scipy and scikit-learn and scikit-image.
The MD application requires pymol to load the trajectory that is not included in the dependencies

To set up the project, install pytorch and run :
\`\`\`
pip install quicksom
\`\`\`
EOF

cat << EOF
#### SOM object interface
The SOM object can be created using any grid size, with a optional periodic topology.
One can also choose optimization parameters such as the number of epochs to train or the batch size

To use it, we include three scripts to fit a SOM, to optionally build
the clusters manually with a gui and to predict cluster affectations
for new data points

EOF

runcmd "quicksom_fit -h"
runcmd "quicksom_gui -h"
runcmd "quicksom_predict -h"

cat << EOF
The SOM object is also importable from python scripts to use
directly in your analysis pipelines.
EOF

cat << EOF
\`\`\`python
import pickle
import numpy
import torch
from som import SOM

# Get data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = numpy.load('contact_desc.npy')
X = torch.from_numpy(X)
X = X.float()
X = X.to(device)

# Create SOM object and train it, then dump it as a pickle object
m, n = 100, 100
dim = X.shape[1]
niter = 5
batch_size = 100
som = SOM(m, n, dim, niter=niter, device=device)
learning_error = som.fit(X, batch_size=batch_size)
som.to_device('cpu')
pickle.dump(som, open('som.pickle', 'wb'))

# Usage on the input data, predicted_clusts is an array of length n_samples with clusters affectations
som = pickle.load(open('som.pickle', 'rb'))
som.to_device(device)
predicted_clusts, errors = som.predict_cluster(X)
\`\`\`
EOF

cat << EOF
#### SOM analysis of molecular dynamics (MD) trajectories.

##### Scripts and extra dependencies:
- \`dcd2npy\`: [Pymol](https://anaconda.org/schrodinger/pymol)
- \`mdx\`: [Pymol](https://anaconda.org/schrodinger/pymol), [pymol-psico](https://github.com/speleo3/pymol-psico)

The SOM algorithm can efficiently map MD trajectories for analysis and clustering purposes.
The script \`dcd2npy\` can be used to select a subset of atoms from a trajectory in \`dcd\` format,
align it and save the selection as a \`npy\` file that can be handled by the command \`quicksom_fit\`.
EOF
runcmd "dcd2npy -h"

# runcmd "./main.py"

# cp -r figs figures

# cat << EOF
# ## Input dataset:
# ![input](https://raw.githubusercontent.com/bougui505/quicksom/master/figures/moons.png)
# ## Umatrix:
# ![Umatrix](https://raw.githubusercontent.com/bougui505/quicksom/master/figures/umat.png)
# ## Data projection:
# ![project](https://raw.githubusercontent.com/bougui505/quicksom/master/figures/project.png)
# ## Cluster projection:
# ![project](https://raw.githubusercontent.com/bougui505/quicksom/master/figures/project_clusts.png)
# ## Cluster affectation:
# ![project](https://raw.githubusercontent.com/bougui505/quicksom/master/figures/clusts.png)
# EOF
