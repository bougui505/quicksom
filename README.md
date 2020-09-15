# Self-Organizing Map
PyTorch implementation of a Self-Organizing Map.
The implementation makes possible the use of a GPU if available for faster computations.
It follows the scikit package semantics for training and usage of the model.

#### Requirements and setup
The SOM object requires torch installed.

It has dependencies in numpy, scipy and scikit-learn and scikit-image.
The MD application requires pymol to load the trajectory that is not included in the dependencies

To set up the project, install pytorch and run :
```
pip install quicksom
```
#### SOM object interface
The SOM object can be created using any grid size, with a optional periodic topology.
One can also choose optimization parameters such as the number of epochs to train or the batch size

To use it, we include three scripts to fit a SOM, to optionally build
the clusters manually with a gui and to predict cluster affectations 
for new data points
```bash
quicksom_fit -h
quicksom_gui -h
quicksom_predict -h
```

The SOM object is also importable from python scripts to use
directly in your analysis pipelines.

```python
import pickle
import numpy
import torch
from quicksom.som import SOM

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
