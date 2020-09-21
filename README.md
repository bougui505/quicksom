# Self-Organizing Map
PyTorch implementation of a Self-Organizing Map.
The implementation makes possible the use of a GPU if available for faster computations.
It follows the scikit package semantics for training and usage of the model.

#### Requirements and setup
The SOM object requires torch installed.

It has dependencies in numpy, scipy and scikit-learn and scikit-image.
The MD application requires pymol to load the trajectory that is not included in the dependencies

To set up the project, we suggest using conda environments.
Install [PyTorch](https://pytorch.org/get-started/locally/) and run :
```
pip install quicksom
```
#### SOM object interface
The SOM object can be created using any grid size, with a optional periodic topology.
One can also choose optimization parameters such as the number of epochs to train or the batch size

To use it, we include three scripts to fit a SOM, to optionally build
the clusters manually with a gui and to predict cluster affectations
for new data points

```
$ quicksom_fit -h

usage: quicksom_fit [-h] -i IN_NAME [-o OUT_NAME] [-m M] [-n N] [--norm NORM]
                    [--periodic] [--n_iter N_ITER] [--batch_size BATCH_SIZE]
                    [--alpha ALPHA] [--sigma SIGMA] [--scheduler SCHEDULER]

optional arguments:
  -h, --help            show this help message and exit
  -i IN_NAME, --in_name IN_NAME
                        name of the .npy to use
  -o OUT_NAME, --out_name OUT_NAME
                        name of pickle to dump
  -m M, --m M           The width of the som
  -n N, --n N           The height of the som
  --norm NORM           The p norm to use
  --periodic            if set, periodic topology is used
  --n_iter N_ITER       The number of iterations
  --batch_size BATCH_SIZE
                        The batch size to use
  --alpha ALPHA         The initial learning rate
  --sigma SIGMA         The initial sigma for the convolution
  --scheduler SCHEDULER
                        Which scheduler to use, can be linear, exp or half
```
```
$ quicksom_gui -h

usage: quicksom_gui [-h] [-i IN_NAME] [-o OUT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  -i IN_NAME, --in_name IN_NAME
                        name of the som to load
  -o OUT_NAME, --out_name OUT_NAME
                        name of the som to dump if we want it different
```
```
$ quicksom_predict -h

usage: quicksom_predict [-h] [-i IN_NAME] [-o OUT_NAME] [-s SOM_NAME]
                        [--recompute_cluster] [--batch BATCH]

optional arguments:
  -h, --help            show this help message and exit
  -i IN_NAME, --in_name IN_NAME
                        name of the npy file to use
  -o OUT_NAME, --out_name OUT_NAME
                        name of txt to dump
  -s SOM_NAME, --som_name SOM_NAME
                        name of pickle to load
  --recompute_cluster   if set, periodic topology is used
  --batch BATCH         Batch size
```
The SOM object is also importable from python scripts to use
directly in your analysis pipelines.
```python
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
```
#### SOM analysis of molecular dynamics (MD) trajectories.

##### Scripts and extra dependencies:
- `dcd2npy`: [Pymol](https://anaconda.org/schrodinger/pymol)
- `mdx`: [Pymol](https://anaconda.org/schrodinger/pymol), [pymol-psico](https://github.com/speleo3/pymol-psico)
```
conda install -c schrodinger pymol pymol-psico
```

The SOM algorithm can efficiently map MD trajectories for analysis and clustering purposes.
The script `dcd2npy` can be used to select a subset of atoms from a trajectory in `dcd` format,
align it and save the selection as a `npy` file that can be handled by the command `quicksom_fit`.
```
$ dcd2npy -h

usage: dcd2npy [-h] --pdb PDB --dcd DCD --select SELECTION

Convert a dcd trajectory to a numpy object

optional arguments:
  -h, --help          show this help message and exit
  --pdb PDB           Topology PDB file
  --dcd DCD           DCD trajectory file
  --select SELECTION  Atoms to select
```
The following commands can be applied for a MD clustering.
- Create a npy file with atomic coordinates of C-alpha:
```
$ dcd2npy --pdb data/2lj5.pdb --dcd data/2lj5.dcd --select 'name CA'

dcdplugin) detected standard 32-bit DCD file of native endianness
dcdplugin) CHARMM format DCD file (also NAMD 2.1 and later)
 ObjectMolecule: read set 1 into state 2...
[...]
 ObjectMolecule: read set 301 into state 302...
 PyMOL not running, entering library mode (experimental)
Coords shape: (301, 228)
```
- Fit the SOM:
```
$ quicksom_fit -i data/2lj5.npy -o data/som_2lj5.p --n_iter 100 --batch_size 50 --periodic --alpha 0.5

1/100: 50/301 | alpha: 0.500000 | sigma: 25.000000 | error: 397.090729 | time 0.387760
4/100: 150/301 | alpha: 0.483333 | sigma: 24.166667 | error: 8.836357 | time 5.738029
7/100: 250/301 | alpha: 0.466667 | sigma: 23.333333 | error: 8.722509 | time 11.213565
[...]
91/100: 50/301 | alpha: 0.050000 | sigma: 2.500000 | error: 5.658005 | time 137.348755
94/100: 150/301 | alpha: 0.033333 | sigma: 1.666667 | error: 5.373021 | time 142.033695
97/100: 250/301 | alpha: 0.016667 | sigma: 0.833333 | error: 5.855451 | time 147.203326
```
- The SOM map can be analyzed and manually cluster using the Graphical User Unterface `quicksom_gui`:
```bash
quicksom_gui -i data/som_2lj5.p
```
- The cluster assignment is performed using:
```
$ quicksom_predict -i data/2lj5.npy -o data/2lj5 -s data/som_2lj5.p

1/42/43/44/4
```
This command generates 3 files:
```
$ ls data/2lj5_*.txt

data/2lj5_bmus.txt
data/2lj5_clusters.txt
data/2lj5_codebook.txt
```
containing the data:
- Best Matching Unit with error for each data point
- Cluster assignment
- Assignment for each SOM cell of the closest data point (BMU with minimal error). `-1` means no assignment
```
$ head -3 data/2lj5_bmus.txt

38.0000 36.0000 4.9054
37.0000 47.0000 4.6754
2.0000 27.0000 7.0854
```
```
$ head -3 data/2lj5_clusters.txt

3 8 21 26 27 31 38 42 43 44 45 47 74 76 77 91 93 97 101 118 125 126 141 146 152 153 161 170 171 179 184 188 189 190 196 205 217 222 225 226 234 244 254 264 284 285 291 298
2 4 6 9 13 20 22 25 28 32 36 50 53 54 62 63 69 73 79 81 82 83 84 85 87 98 102 103 105 106 107 115 120 122 128 130 131 132 138 139 145 147 149 154 158 160 162 164 169 172 178 180 182 199 208 213 216 219 220 227 228 230 236 238 239 240 246 247 249 250 255 257 259 266 274 276 277 278 286 290 292 295 296 300
0 1 7 10 11 12 14 16 17 18 19 23 24 29 30 34 37 40 41 49 51 55 57 59 60 61 64 65 67 68 70 71 72 78 86 88 89 90 92 94 95 96 100 104 108 109 111 112 113 117 119 121 123 124 129 133 135 136 137 140 142 143 144 150 151 155 156 157 159 165 167 168 173 174 175 176 177 183 186 187 192 194 200 204 207 209 210 211 212 214 215 221 224 229 231 232 233 235 241 243 245 248 251 252 253 258 260 261 263 265 267 269 270 271 273 275 279 281 282 283 287 288 289 294 297 299
```
