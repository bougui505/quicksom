#!/usr/bin/env python3

import pickle
import numpy
import torch
from quicksom.som import SOM

import argparse

parser = argparse.ArgumentParser()
# In/Out
parser.add_argument("-i", "--in_name", default=None, help="name of the txt to use")
parser.add_argument("-o", "--out_name", default='out.txt', help="name of txt to dump")
parser.add_argument("-s", "--som_name", default='som.p', help="name of pickle to load")
parser.add_argument("--recompute_cluster", default=False, action='store_true', help="if set, periodic topology is used")
args, _ = parser.parse_known_args()

som = pickle.load(open(args.som_name, 'rb'))
if args.recompute_cluster:
    som.cluster_att = None

if args.in_name is not None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = numpy.genfromtxt(args.in_name)
    # y = X[:, 2]
    # X = X[:, :2]
    X = torch.from_numpy(X)
    X = X.float()
    X = X.to(device)
    predicted_clusts, errors = som.predict_cluster(X)
else:
    predicted_clusts, errors = som.predict_cluster()

# numpy.savetxt(args.out_name, predicted_clusts, fmt='%d')
with open(args.out_name, 'w') as outfile:
    for cid in numpy.unique(predicted_clusts):
        inds = numpy.where(predicted_clusts == cid)[0]
        outfile.write(f"{' '.join(['%d' % e for e in inds])}\n")
