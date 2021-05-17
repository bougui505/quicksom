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

import pickle
import argparse
import numpy
import scipy.ndimage
import torch
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max


class Click:
    def __init__(self, ax):
        self.ax = ax
        self.pos = (0, 0)
        self.clickpos, = ax.plot(self.pos[1], self.pos[0], c='r', marker='s')

    def __call__(self, event):
        self.pos = int(event.ydata), int(event.xdata)
        # print(f'i={self.pos[0]}; j={self.pos[1]}')
        self.clickpos.set_data(self.pos[1], self.pos[0])
        self.clickpos.figure.canvas.draw()


def clean_contours(contourplot):
    if contourplot is not None:
        for coll in contourplot.collections:
            coll.remove()


class Wheel:
    def __init__(self, som, click, ax):
        self.ax = ax
        self.threshold = 0
        self.precision = .01
        self.display_str = 'Threshold=%.3f'
        help_string = "\n\n- Left-Click: Set position\n- Scroll: Set basin threshold\n- Shift-Scroll: Set basin threshold with higher precision\n- Left-Click on basin: Set the current basin as a cluster\n- Right-Click on cluster: Delete the selected cluster\n- Double-Click on cluster: Expand the clusters\n- Left-Click on cluster: Come back to user cluster view"
        self.display_str += help_string
        self.threshold_display = self.ax.text(0.5, -0.1, self.display_str % self.threshold,
                                              ha="center", va='top', transform=self.ax.transAxes)
        self.som = som
        self.click = click
        self.pos = self.click.pos
        self.clusters = numpy.zeros((self.som.m, self.som.n), dtype=int)
        self.expanded_clusters = numpy.zeros((self.som.m, self.som.n), dtype=int)
        self.cluster_current = numpy.zeros((self.som.m, self.som.n), dtype=bool)
        self.clusterplot = None  # To plot the current cluster
        self.clustersplot = self.ax.scatter(0, 0, c='r', marker='s', alpha=0.)  # To plot all the clusters

        self.local_min = peak_local_max(-self.som.uumat, min_distance=1)
        self.n_local_min = self.local_min.shape[0]
        self.local_min = numpy.asarray([self.som.reversed_mapping[(e[0], e[1])] for e in self.local_min])
        plt.scatter(self.local_min[:, 1], self.local_min[:, 0], c='g')

    def plot_clusters(self, plot_expanded=False):
        if plot_expanded:
            to_plot = self.expanded_clusters
        else:
            to_plot = self.clusters
        to_plot = numpy.ma.masked_array(data=to_plot, mask=(to_plot == 0))
        self.clustersplot.remove()
        self.clustersplot = self.ax.matshow(to_plot, cmap='tab20')
        self.clustersplot.figure.canvas.draw()

    def __call__(self, event):
        if event.key is 'shift':
            self.precision = 0.001
        else:
            self.precision = 0.01
        if event.button is 'up':
            self.threshold += self.precision
            if self.threshold > 1.:
                self.threshold = 1.
        if event.button is 'down':
            self.threshold -= self.precision
            if self.threshold < 0.:
                self.threshold = 0.
        if event.button == 1:  # Left mouse button pressed
            if self.clusters[self.click.pos] == 0:
                cluster_id = self.clusters.max() + 1
                # print(f"Creating cluster {cluster_id}")
                self.clusters[self.cluster_current] = cluster_id
                self.remap_clusters()
                self.expand_clusters()
            self.plot_clusters()
            if event.dblclick:
                self.plot_clusters(plot_expanded=True)
        if event.button == 3:  # Right mouse button pressed
            cluster_id = self.clusters[self.click.pos]
            if cluster_id > 0:
                self.delete_cluster(cluster_id)
                self.expand_clusters()
                self.plot_clusters()
        self.threshold_display.set_text(self.display_str % self.threshold)
        self.cluster()
        plt.draw()

    @property
    def is_cluster(self):
        return self.cluster_current[self.click.pos]

    def delete_cluster(self, cluster_id):
        # print(f"Deleting cluster {cluster_id}")
        self.clusters[self.clusters == cluster_id] = 0
        self.remap_clusters()

    @property
    def cluster_ids(self):
        cluster_ids = numpy.unique(self.clusters)
        cluster_ids = cluster_ids[cluster_ids > 0]
        return cluster_ids

    def remap_clusters(self):
        remap = numpy.zeros_like(self.clusters)
        for i, cid in enumerate(self.cluster_ids):
            sel = (self.clusters == cid)
            remap[sel] = i + 1
        self.clusters = remap
        print(f'Clusters: {numpy.unique(self.cluster_ids)}')

    def cluster(self):
        pos = self.click.pos
        # ind = numpy.ravel_multi_index(pos, (self.som.m, self.som.n))
        # dists = self.som.all_to_all_dist[ind][numpy.ravel_multi_index(self.local_min.T, (self.som.m, self.som.n))]
        # pos = tuple(self.local_min[numpy.argmin(dists)])
        if self.pos != pos:
            self.threshold = self.som.umat[pos]
        uclusters = self.som.uumat < self.threshold
        label, num_features = scipy.ndimage.label(uclusters)
        label_id = label[self.som.mapping[pos]]
        clusters = numpy.zeros_like(self.som.umat, dtype=bool)
        if label_id > 0:
            zone = (label == label_id)
            uclusters = zone
            for i in range(self.som.m):
                for j in range(self.som.n):
                    u, v = self.som.mapping[(i, j)]
                    clusters[i, j] = uclusters[u, v]
        clean_contours(self.clusterplot)
        self.clusterplot = self.ax.contour(clusters, levels=1, colors='w')
        self.cluster_current = numpy.copy(clusters)
        self.pos = pos

    def expand_clusters(self):
        n_clusters = len(self.cluster_ids)
        expanded_clusters = numpy.copy((self.clusters)).flatten()
        if n_clusters > 1:
            voidcells = numpy.where((self.clusters == 0).flatten())[0]
            for cell in voidcells:
                neighbors = self.som.all_to_all_dist[cell].argsort()
                neighbor_clusters = self.clusters.flatten()[neighbors]
                cluster_assigned = neighbor_clusters[neighbor_clusters > 0][0]
                expanded_clusters[cell] = cluster_assigned
        self.expanded_clusters = expanded_clusters.reshape((self.som.m, self.som.n))

    def format_coord(self, x, y):
        return f'i={int(y)}, j={int(x)}, cluster {self.expanded_clusters[int(y), int(x)]}'


if __name__ == '__main__':
    from quicksom.som import SOM
    parser = argparse.ArgumentParser(description='SOM Graphical User Interface')
    parser.add_argument('--som', type=str, help='SOM pickle object file',
                        required=True)
    args = parser.parse_args()
    sompickle = args.som
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    som = pickle.load(open(sompickle, 'rb'))
    som.to_device(device)
    # som.cluster()

    fig, ax = plt.subplots(figsize=(8, 10))
    cax = ax.matshow(som.umat)
    fig.colorbar(cax)

    click = Click(ax=ax)
    fig.canvas.mpl_connect('button_press_event', click)
    wheel = Wheel(som, click, ax=ax)
    if som.clusters_user is not None:
        wheel.clusters = som.clusters_user
        wheel.expand_clusters()
        wheel.plot_clusters()
    fig.canvas.mpl_connect('scroll_event', wheel)
    fig.canvas.mpl_connect('button_press_event', wheel)
    ax.format_coord = wheel.format_coord
    plt.show()
    som.cluster_att = wheel.expanded_clusters.flatten()
    som.clusters_user = wheel.clusters
    som.to_device('cpu')
    pickle.dump(som, open(sompickle, 'wb'))
