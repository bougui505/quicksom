#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
#                 				                            #
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

import streamlit as st
import dill as pickle
import torch
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


class CPU_Unpickler(pickle.Unpickler):
    """
    Usage:
    f = open(filename, 'rb')
    contents = CPU_Unpickler(f).load()
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_som(picklefile):
    som = CPU_Unpickler(picklefile).load()
    return som


def crop_umat(som, vmin=None, vmax=None):
    umat = som.umat.copy()
    if vmin is None:
        vmin = umat.min()
    if vmax is None:
        vmax = umat.max()
    umat = np.clip(a=umat, a_min=vmin, a_max=vmax)
    return umat


def crop_mds(som, vmin, vmax):
    x, y = som.mds.T
    z = crop_umat(som, vmin, vmax).flatten()
    return x, y, z


def stringfilter(string, threshold, arr):
    """
    see: https://stackoverflow.com/a/38986394/1679629
    """
    def matchdist(tosearch, string):
        out = re.search(tosearch, string)
        if out is not None:
            dist = 1. - len(out.group()) / len(tosearch)
        else:
            dist = 1.
        return dist

    dists = np.asarray([matchdist(string, e) for e in arr])
    sel = dists <= threshold
    return sel


def filter_bmus(som, text, threshold=0.):
    labels = som.labels
    sel = stringfilter(string=text, threshold=threshold, arr=labels)
    return sel


if __name__ == '__main__':

    # -- Set page config
    apptitle = 'QuickSOM app'
    st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

    # Title the app
    st.title('QuickSOM app')
    uploaded_file = st.file_uploader(label='SOM pickle file to load')
    if uploaded_file is not None:
        som = load_som(uploaded_file)

        # -- Create sidebar for plot controls
        st.sidebar.markdown('## Set Plot Parameters')
        vmin = float(som.umat.min())
        vmax = float(som.umat.max())
        vmin_setter = st.sidebar.slider('vmin', vmin, vmin + (vmax - vmin) / 2., vmin)  # min, max, default
        vmax_setter = st.sidebar.slider('vmax', vmin + (vmax - vmin) / 2., vmax, vmax)  # min, max, default

        # Text input to filter bmus
        instr = st.text_input(label='Filter BMUs based on label')
        bmu_selection = None
        if instr is not None and len(instr) > 0:
            bmu_selection = filter_bmus(som, instr)
            nmatch = bmu_selection.sum()
            st.write(f'{nmatch} data match')

        umat = crop_umat(som, vmin=vmin_setter, vmax=vmax_setter)
        mds = crop_mds(som, vmin=vmin_setter, vmax=vmax_setter)
        fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
        img = axs[0].matshow(umat, cmap='coolwarm', aspect='auto')
        axs[1].scatter(mds[0], mds[1], c=mds[2], cmap='coolwarm')
        if bmu_selection is not None:
            selected_cells = som.bmus[bmu_selection]
            axs[0].scatter(selected_cells[:, 1], selected_cells[:, 0], color='cyan')
            selected_mds = som.mds[np.ravel_multi_index(tuple(selected_cells.T), (som.m, som.n))]
            axs[1].scatter(selected_mds[:, 1], selected_mds[:, 0], color='cyan')
        fig.colorbar(img, ax=axs.ravel().tolist())
        st.pyplot(fig=fig)

        bmus = pd.DataFrame(data={
            'bmu_i': som.bmus[:, 0],
            'bmu_j': som.bmus[:, 1],
            'label': som.labels
        },
                            index=range(len(som.bmus)))
