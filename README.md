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
