#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Authors: Vincent Mallet, Guillaume Bouvier	      	                    #
# Copyright (c) 2021 Institut Pasteur                                       #
#                 				                                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################

import os
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from quicksom.som import SOM

from sklearn.datasets import make_moons

try:
    os.mkdir('out')
except FileExistsError:
    pass

try:
    os.mkdir('figs')
except FileExistsError:
    pass

# BUILD DATASET if does not exist yet or too short for the required number of points
max_points = 100
create_data = False
if os.path.exists('data/moons.txt'):
    X = np.genfromtxt('data/moons.txt')
    if len(X) >= max_points:
        create_data = False
        y = X[:, 2]
        X = X[:, :2]

if create_data:
    X, y = make_moons(n_samples=max_points, noise=0.05)
    np.savetxt('data/moons.txt', np.c_[X, y])

X = X[:max_points]
y = y[:max_points]

plt.scatter(X[:, 0], X[:, 1])
plt.savefig('figs/moons.png')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = torch.from_numpy(X)
X = X.float()
X = X.to(device)

if not os.path.exists('out/trained.p'):
    m, n = 50, 50
    dim = X.shape[1]
    niter = 100
    batch_size = 100
    som = SOM(m, n, dim, niter=niter, device=device)
    learning_error = som.fit(X, batch_size=batch_size)
    bmus, inference_error = som.predict(X, batch_size=batch_size)
else:
    pass
    som = pickle.load(open('out/trained.p', 'rb'))

som.to_device('cpu')
pickle.dump(som, open('out/trained.p', 'wb'))
X = X.to('cpu')
bmus, inference_error = som.predict(X)
smap = som.centroids.cpu().numpy().reshape((som.m, som.n, -1))
predicted_clusts, errors = som.predict_cluster(X)
umat = som.umat
plt.matshow(umat)
plt.colorbar()
plt.savefig('figs/umat.png')
plt.clf()

# Problem in the x,y scale formalism vs the grid format. First coordinate of an array is the rows...
plt.matshow(umat)
plt.scatter(bmus[:, 1][y == 0], bmus[:, 0][y == 0], c='red')
plt.scatter(bmus[:, 1][y == 1], bmus[:, 0][y == 1], c='green')
plt.savefig('figs/project.png')
plt.clf()

# Problem in the x,y scale formalism vs the grid format. First coordinate of an array is the rows...
plt.matshow(umat)
plt.scatter(bmus[:, 1], bmus[:, 0], c=predicted_clusts)
plt.savefig('figs/project_clusts.png')
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=predicted_clusts)
plt.savefig('figs/clusts.png')
plt.clf()
