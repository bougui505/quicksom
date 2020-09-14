#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-09-11 08:55:38 (UTC+0200)

import pickle
import argparse
import numpy
import scipy.ndimage
import torch
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

parser = argparse.ArgumentParser(description='SOM Graphical User Interface')
parser.add_argument('--som', type=str, help='SOM pickle object file',
                    required=True)
args = parser.parse_args()


class Click:
    def __init__(self):
        self.pos = (0, 0)
        self.clickpos, = ax.plot(self.pos[1], self.pos[0], c='r', marker='s')

    def __call__(self, event):
        self.pos = int(event.ydata), int(event.xdata)
        print(f'i={self.pos[0]}; j={self.pos[1]}')
        self.clickpos.set_data(self.pos[1], self.pos[0])
        self.clickpos.figure.canvas.draw()


def clean_contours(contourplot):
    if contourplot is not None:
        for coll in contourplot.collections:
            coll.remove()


class Wheel:
    def __init__(self, som, click):
        self.threshold = 0
        self.precision = .01
        self.display_str = 'Threshold=%.3f'
        self.threshold_display = ax.text(0.5, -0.1, self.display_str % self.threshold,
                                         ha="center", transform=ax.transAxes)
        self.som = som
        self.click = click
        self.pos = self.click.pos
        self.clusters = numpy.zeros((self.som.m, self.som.n), dtype=int)
        self.cluster_current = numpy.zeros((self.som.m, self.som.n), dtype=bool)
        self.clusterplot = None  # To plot the current cluster
        self.clustersplot = []  # To plot all the clusters

        self.local_min = peak_local_max(-self.som.uumat, min_distance=1)
        self.n_local_min = self.local_min.shape[0]
        self.local_min = numpy.asarray([self.som.reversed_mapping[(e[0], e[1])] for e in self.local_min])
        plt.scatter(self.local_min[:, 1], self.local_min[:, 0], c='g')

    def plot_clusters(self):
        for cplot in self.clustersplot:
            clean_contours(cplot)
        self.clustersplot = []
        for cid in self.cluster_ids:
            cplot = ax.contour(self.clusters == cid, levels=1, colors='r')
            self.clustersplot.append(cplot)

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
            if self.is_cluster:
                cluster_id = self.clusters.max() + 1
                print(f"Creating cluster {cluster_id}")
                self.clusters[self.cluster_current] = cluster_id
                self.remap_clusters()
                self.plot_clusters()
        if event.button == 3:  # Right mouse button pressed
            cluster_id = self.clusters[self.click.pos]
            if cluster_id > 0:
                self.delete_cluster(cluster_id)
                self.plot_clusters()
        self.threshold_display.set_text(self.display_str % self.threshold)
        self.cluster()
        plt.draw()

    @property
    def is_cluster(self):
        return self.cluster_current[self.click.pos]

    def delete_cluster(self, cluster_id):
        print(f"Deleting cluster {cluster_id}")
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
        ind = numpy.ravel_multi_index(pos, (self.som.m, self.som.n))
        dists = self.som.all_to_all_dist[ind][numpy.ravel_multi_index(self.local_min.T, (self.som.m, self.som.n))]
        pos = tuple(self.local_min[numpy.argmin(dists)])
        if self.pos != pos:
            self.threshold = self.som.umat[pos] + 10. * self.precision
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
        self.clusterplot = ax.contour(clusters, levels=1, colors='w')
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
        self.clusters = expanded_clusters.reshape((self.som.m, self.som.n))


def format_coord(x, y):
    return f'i={int(y)}, j={int(x)}'


if __name__ == '__main__':
    sompickle = args.som
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    som = pickle.load(open(sompickle, 'rb'))
    som.to_device(device)
    som.cluster()

    fig, ax = plt.subplots()
    ax.format_coord = format_coord
    cax = ax.matshow(som.umat)
    fig.colorbar(cax)

    click = Click()
    fig.canvas.mpl_connect('button_press_event', click)
    wheel = Wheel(som, click)
    fig.canvas.mpl_connect('scroll_event', wheel)
    fig.canvas.mpl_connect('button_press_event', wheel)
    plt.show()
