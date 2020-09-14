#!/usr/bin/env python3

import pickle
import numpy
import torch

import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from quicksom import SOM

import argparse

parser = argparse.ArgumentParser()
# In/Out
parser.add_argument("-in_name", "--in_name", default=None, help="name of the txt to use")
parser.add_argument("-out_name", "--out_name", default='out.npy', help="name of npy to dump")
parser.add_argument("-som_name", "--som_name", default='som.p', help="name of pickle to load")
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

numpy.save(args.out_name, predicted_clusts)
