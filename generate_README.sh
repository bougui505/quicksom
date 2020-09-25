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

func runcmd_cut() {
    OUTPUT=$(eval $1)
    echo "\`\`\`"
    echo "$ $1\n"
    echo "$OUTPUT" | head -3
    echo "[...]"
    echo "$OUTPUT" | tail -3
    echo "\`\`\`"
}

func runcmd_null() {
    OUTPUT=$(eval $1)
    echo "\`\`\`bash"
    echo "$1\n"
    echo "\`\`\`"
}

[ -d figures ] && rm -r figures

cat << EOF
# Self-Organizing Map
PyTorch implementation of a Self-Organizing Map.
The implementation makes possible the use of a GPU if available for faster computations.
It follows the scikit package semantics for training and usage of the model.
It also includes runnable scripts to avoid coding.

EOF

cat << EOF
### Requirements and setup
The SOM object requires PyTorch installed.

It has dependencies in numpy, scipy and scikit-learn and scikit-image.
The MD application requires pymol to load the trajectory that is not included in the dependencies

To set up the project, we suggest using conda environments.
Install [PyTorch](https://pytorch.org/get-started/locally/) and run :
\`\`\`
pip install quicksom
\`\`\`
EOF

cat << EOF
### SOM object interface
The SOM object can be created using any grid size, with a optional periodic topology.
One can also choose optimization parameters such as the number of epochs to train or the batch size

To use it, we include three scripts to :
 - fit a SOM
 - to optionally build the clusters manually with a gui
 - to predict cluster affectations for new data points

EOF

runcmd "quicksom_fit -h"
runcmd "quicksom_gui -h"
runcmd "quicksom_predict -h"

cat << EOF
The SOM object is also importable from python scripts to use
directly in your analysis pipelines :
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
### SOM training on molecular dynamics (MD) trajectories

#### Scripts and extra dependencies:
- \`dcd2npy\`: [Pymol](https://anaconda.org/schrodinger/pymol)
- \`mdx\`: [Pymol](https://anaconda.org/schrodinger/pymol), [pymol-psico](https://github.com/speleo3/pymol-psico)

To set these dependencies up using conda, just type :
\`\`\`
conda install -c schrodinger pymol pymol-psico
\`\`\`

The SOM algorithm can efficiently map MD trajectories for analysis and clustering purposes.
The script \`dcd2npy\` can be used to select a subset of atoms from a trajectory in \`dcd\` format,
align it and save the selection as a \`npy\` file that can be handled by the command \`quicksom_fit\`.
EOF
runcmd "dcd2npy -h"

cat << EOF
The following commands can be applied for a MD clustering.
#### Create a npy file with atomic coordinates of C-alpha:
EOF
runcmd_cut "dcd2npy --pdb data/2lj5.pdb --dcd data/2lj5.dcd --select 'name CA'"
cat << EOF
#### Fit the SOM:
EOF
if [ -f data/som_2lj5.p ]; then
    cat << EOF
\`\`\`
$ quicksom_fit -i data/2lj5.npy -o data/som_2lj5.p --n_iter 100 --batch_size 50 --periodic --alpha 0.5

1/100: 50/301 | alpha: 0.500000 | sigma: 25.000000 | error: 397.090729 | time 0.387760
4/100: 150/301 | alpha: 0.483333 | sigma: 24.166667 | error: 8.836357 | time 5.738029
7/100: 250/301 | alpha: 0.466667 | sigma: 23.333333 | error: 8.722509 | time 11.213565
[...]
91/100: 50/301 | alpha: 0.050000 | sigma: 2.500000 | error: 5.658005 | time 137.348755
94/100: 150/301 | alpha: 0.033333 | sigma: 1.666667 | error: 5.373021 | time 142.033695
97/100: 250/301 | alpha: 0.016667 | sigma: 0.833333 | error: 5.855451 | time 147.203326
\`\`\`
EOF
else
runcmd_cut "quicksom_fit -i data/2lj5.npy -o data/som_2lj5.p --n_iter 100 --batch_size 50 --periodic --alpha 0.5"
fi

cat << EOF

#### Analysis and clustering of the map using \`quicksom_gui\`:
\`\`\`bash
quicksom_gui -i data/som_2lj5.p
\`\`\`

### Analysis of MD trajectories with this SOM
We now have a trained SOM and we can use several functionalities such as clustering input data points and grouping them
into separate dcd files, creating a dcd with one centroid per fram or plotting of the U-Matrix and its flow.

#### Cluster assignment of input data points:
EOF
runcmd_null "quicksom_predict -i data/2lj5.npy -o data/2lj5 -s data/som_2lj5.p"
cat << EOF
This command generates 3 files:
EOF
runcmd "ls data/2lj5_*.txt"
cat << EOF
containing the data:
    - Best Matching Unit with error for each data point
    - Cluster assignment
    - Assignment for each SOM cell of the closest data point (BMU with minimal error). \`-1\` means no assignment
EOF
runcmd "head -3 data/2lj5_bmus.txt"
runcmd "head -3 data/2lj5_clusters.txt"
cat << EOF
#### Cluster extractions from the input \`dcd\` using the \`quicksom_extract\` tool:
EOF
runcmd 'quicksom_extract -h'
runcmd_null 'quicksom_extract -p data/2lj5.pdb -t data/2lj5.dcd -c data/2lj5_clusters.txt'
runcmd "ls -v data/cluster_*.dcd"
cat << EOF
#### Extraction of the SOM centroids from the input \`dcd\`
EOF
runcmd_null 'grep -v "\-1" data/2lj5_codebook.txt > _codebook.txt
mdx --top data/2lj5.pdb --traj data/2lj5.dcd --fframes _codebook.txt --out data/centroids.dcd
rm _codebook.txt'
cat << EOF
#### Plotting the U-matrix:
EOF
runcmd_null "python3 -c 'import pickle
import matplotlib.pyplot as plt
som=pickle.load(open(\"data/som_2lj5.p\", \"rb\"))
plt.matshow(som.umat)
plt.savefig(\"data/umat_2lj5.png\")
'"
cat << EOF
#### Flow analysis
The flow of the trajectory can be projected onto the U-matrix using the following command:
EOF
runcmd "quicksom_flow -h"
