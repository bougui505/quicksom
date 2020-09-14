#!/usr/bin/env python3

import pickle
from quicksom.som import SOM

import argparse

parser = argparse.ArgumentParser()
# In/Out
parser.add_argument("-i", "--in_name", default='som.p', help="name of the som to load")
parser.add_argument("-o", "--out_name", default=None, help="name of the som to dump if we want it different")
args, _ = parser.parse_known_args()

som = pickle.load(open(args.in_name, 'rb'))
som.manual_cluster()
out_name = args.in_name if args.out_name is None else args.out_name

pickle.dump(som, open(out_name, 'wb'))
