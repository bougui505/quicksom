#!/usr/bin/env python3

import pickle
import numpy
import torch

from quicksom.som import SOM

import argparse

parser = argparse.ArgumentParser()
# In/Out
parser.add_argument("-i", "--in_name", required=True, help="name of the .npy to use")
parser.add_argument("-o", "--out_name", default='som.p', help="name of pickle to dump")
# SOM
parser.add_argument("-m", "--m", type=int, default=50, help="The width of the som")
parser.add_argument("-n", "--n", type=int, default=50, help="The height of the som")
parser.add_argument("--norm", type=int, default=2, help="The p norm to use")
parser.add_argument("--periodic", default=False, action='store_true', help="if set, periodic topology is used")
# Optim
parser.add_argument("--n_iter", type=int, default=30, help="The number of iterations")
parser.add_argument("--batch_size", type=int, default=100, help="The batch size to use")
parser.add_argument("--alpha", type=float, default=None, help="The initial learning rate")
parser.add_argument("--sigma", type=float, default=None, help="The initial sigma for the convolution")
parser.add_argument("--scheduler", default='linear', help="Which scheduler to use, can be linear, exp or half")
args, _ = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = numpy.load(args.in_name)

X = torch.from_numpy(X)
X = X.float()
X = X.to(device)
dim = X.shape[1]
som = SOM(args.m,
          args.n,
          dim,
          alpha=args.alpha,
          sigma=args.sigma,
          sched=args.scheduler,
          precompute=True,
          periodic=args.periodic,
          p_norm=args.norm,
          niter=args.n_iter,
          device=device)

learning_error = som.fit(X, batch_size=args.batch_size)
som.to_device('cpu')
pickle.dump(som, open(args.out_name, 'wb'))
