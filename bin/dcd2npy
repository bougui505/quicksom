#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-08-28 10:34:16 (UTC+0200)

import os
import pymol.cmd as cmd
import numpy
import argparse

parser = argparse.ArgumentParser(description='Convert a dcd trajectory to a numpy object')
parser.add_argument('--pdb', dest='pdb', type=str,
                    help='Topology PDB file', required=True)
parser.add_argument('--dcd', dest='dcd', type=str,
                    help='DCD trajectory file', required=True)
parser.add_argument('--select', dest='selection', type=str,
                    help='Atoms to select', required=True)
args = parser.parse_args()

cmd.load(args.pdb, object='mystructure')
cmd.load_traj(args.dcd, state=0)
rmsds = cmd.intra_fit(args.selection)
natoms = cmd.select(args.selection)
nstates = cmd.count_states('mystructure')
coords = cmd.get_coords(args.selection, state=0)
coords = coords.reshape((nstates, natoms * 3))
coords = coords[1:]
print(f'Coords shape: {coords.shape}')
numpy.save(f'{os.path.splitext(args.dcd)[0]}.npy', coords)
