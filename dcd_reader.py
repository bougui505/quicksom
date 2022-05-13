#!/usr/bin/env python
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

import argparse
import os
import itertools
import functools
import numpy as np
import time

import torch

import MDAnalysis.coordinates.DCD as md
import pymol.cmd as cmd


def pymol_process(pdb, selection='all', select_align=None):
    """
    Use the PDB file to find the indices in the array for the SOM fitting and the frame alignment.
    :param pdb:
    :param selection:
    :param select_align:
    :return:
    """
    cmd.set('retain_order', 1)
    cmd.load(pdb, object='mystructure')
    pymolspace = {'atoms_ids': [], 'align_ids': []}
    cmd.iterate(selection, 'atoms_ids.append(rank)', space=pymolspace)
    atoms_ids = np.asarray(pymolspace['atoms_ids'])

    # Optionnaly use different indices for alignment.
    if select_align is not None:
        cmd.iterate(select_align, 'align_ids.append(rank)', space=pymolspace)
        align_ids = np.asarray(pymolspace['align_ids'])
    else:
        align_ids = atoms_ids
    return atoms_ids, align_ids


def rigid_body_fit(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Numpy array of shape (N,D) -- Point Cloud to Align (source)
        -    B: Numpy array of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.dot(B_c)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    # Rotation matrix
    R = V.dot(U.T)
    # Translation vector
    t = b_mean - R.dot(a_mean)
    return R, t


def transform(coords, R, t):
    """
    Apply R and a translation t to coords
    """
    coords_out = R.dot(coords.T).T + t
    return coords_out


class DCDataset(torch.utils.data.Dataset):
    def __init__(self, pdbfilename, dcdfilename, selection='polymer', selection_alignment=None):
        """
        This dataset opens a dcd file and make it a PyTorch dataset
        :param dcdfilename:
        """

        self.dcdfilename = dcdfilename
        self.dcdfile = None
        self.fastafile = None
        atoms_ids, align_ids = pymol_process(pdb=pdbfilename, selection=selection, select_align=selection_alignment)
        self.atoms_ids, self.align_ids = atoms_ids, align_ids
        with md.DCDFile(dcdfilename) as dcd:
            self.len = len(dcd)
            first_frame = dcd.read()
            coords = first_frame.xyz
            sel_coords = coords[atoms_ids]
            self.dim = sel_coords.flatten().size
            self.reference_coords = coords[align_ids]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        Make each worker have a copy of the dcdfile.
        Then for each fetch, use seek to get the right frame, extract the relevant indices.
        Finally use the select_align ones to align to the reference frame and rotate the selected indices accordingly.
        :param index:
        :return:
        """
        if self.dcdfile is None:
            # print(f'opening {self.dcdfilename} with worker id :'
            #       f' {torch.utils.data.get_worker_info().id} and obj id :'
            #       f' {id(self.dcdfilename)} ')
            self.dcdfile = md.DCDFile(self.dcdfilename)

        self.dcdfile.seek(index)
        frame = self.dcdfile.read()
        coords = frame.xyz

        align_coords = coords[self.align_ids]
        R, t = rigid_body_fit(align_coords, self.reference_coords)
        sel_coords = coords[self.atoms_ids]
        aligned_sel_coords = transform(sel_coords, R, t)
        flat_coords = aligned_sel_coords.flatten()
        assert flat_coords.size == self.dim
        return index, flat_coords


def test_parallel(dataset, num_workers, batch_size=10, nloop=100):
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    t0 = time.time()
    dataiter = itertools.cycle(dataloader)
    for i in range(nloop):
        frame_idx, inputvectors = next(dataiter)
    deltat = time.time() - t0
    print(f'Timing for {num_workers} worker(s), batch_size {batch_size} and {nloop} loops: {deltat:.3f} s')
    return deltat


if __name__ == '__main__':
    pass
    pdbname = "data/2lj5.pdb"
    dcdname = "data/2lj5.dcd"
    dataset = DCDataset(dcdfilename=dcdname, pdbfilename=pdbname, selection='name CA')
    test_parallel(num_workers=2, dataset=dataset)
