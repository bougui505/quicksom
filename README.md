# Self-Organizing Map
PyTorch implementation of a Self-Organizing Map.
The implementation makes possible the use of a GPU if available for faster computations.
It follows the scikit package semantics for training and usage of the model.

# Requirements
The SOM object requires numpy, scipy and torch installed.

The graph-based clustering requires scikit-learn and the image-based clustering requires scikit-image. By default,
we use the graph-based clustering

The toy example uses scikit-learn for the toy dataset generation

The MD application requires pymol for loading the trajectory

Then one can run :
```
pip install quicksom
```
# SOM object interface
The SOM object can be created using any grid size, with a optional periodic topology.
One can also choose optimization parameters such as the number of epochs to train or the batch size
```python
import pickle
import numpy
import torch
from som import SOM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = numpy.load('contact_desc.npy')
X = torch.from_numpy(X)
X = X.float()
X = X.to(device)
m, n = 100, 100
dim = X.shape[1]
niter = 5
batch_size = 100
som = SOM(m, n, dim, niter=niter, device=device)
learning_error = som.fit(X, batch_size=batch_size)
bmus, inference_error = som.predict(X, batch_size=batch_size)
predicted_clusts, errors = som.predict_cluster(X)
som.to_device('cpu')
pickle.dump(som, open('som.pickle', 'wb'))
```
```
$ ./main.py

training ... cpu
_parameters ... cpu
_buffers ... cpu
_non_persistent_buffers_set ... cpu
_backward_hooks ... cpu
_forward_hooks ... cpu
_forward_pre_hooks ... cpu
_state_dict_hooks ... cpu
_load_state_dict_pre_hooks ... cpu
_modules ... cpu
m ... cpu
n ... cpu
grid_size ... cpu
dim ... cpu
periodic ... cpu
p_norm ... cpu
sched ... cpu
niter ... cpu
alpha ... cpu
sigma ... cpu
centroids -> cpu
locations -> cpu
maprange -> cpu
offset1 -> cpu
offset2 -> cpu
offset3 -> cpu
offset4 -> cpu
offset5 -> cpu
offset6 -> cpu
offset7 -> cpu
offset8 -> cpu
precompute ... cpu
distance_mat -> cpu
umat ... cpu
cluster_att ... cpu
alpha_op ... cpu
sigma_op ... cpu
```
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
