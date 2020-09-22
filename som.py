#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-07-02 15:02:22 (UTC+0200)

import sys
import torch
import torch.nn as nn
import time
import itertools
import matplotlib.pyplot as plt

import numpy as np
import scipy.spatial
import scipy.sparse

from scipy import ndimage as ndi
import scipy.sparse.csgraph as graph

from skimage.feature import peak_local_max

# import skimage
# from skimage.morphology import disk
# from skimage.filters import rank
# from skimage.segmentation import watershed
# from skimage.segmentation import morphological_geodesic_active_contour, felzenszwalb, slic
# from skimage.feature import peak_local_max
# from skimage import filters

from sklearn.cluster import AgglomerativeClustering


class SOM(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.

    Creates a grid of size m*n with centroids of dimensions dim randomly inititalised.
    precompute is longer to initialize but runs significantly faster
    use device to use a gpu
    Use device to use a gpu
    :param m: the height of the SOM
    :param n: the width of the SOM
    :param dim: the dimension of the input vectors
    :param alpha: The initial lr
    :param sigma: The initial width of the gaussian blur. If None, half of the grid is used
    :param niter: The number of epochs to run the SOM
    :param sched: Scheduler scheme. Possibilites are 'linear' for a linear decay, 'half' to half the lr 20 times and
                    'exp' for an exponential decay
    :param device: the torch device to create the SOM onto. This can be modified using the to() method
    :param precompute: Speedup for initialization. Creates a little overhead for very small trainings
    :param periodic: Boolean to use a periodic topology
    :param p_norm: p value for the p-norm distance to calculate between each vector pair for torch.cdist
     """

    def __init__(self, m, n, dim,
                 alpha=None,
                 sigma=None,
                 niter=2,
                 sched='linear',
                 device='cpu',
                 precompute=True,
                 periodic=False,
                 p_norm=2):

        # topology of the som
        super(SOM, self).__init__()
        self.m = m
        self.n = n
        self.grid_size = m * n
        self.dim = dim
        self.periodic = periodic

        # optimization parameters
        self.p_norm = p_norm
        self.sched = sched
        self.niter = niter
        if alpha is not None:
            self.alpha = float(alpha)
        else:
            self.alpha = alpha
        if sigma is None:
            self.sigma = np.sqrt(self.m * self.n) / 2.0
        else:
            self.sigma = float(sigma)

        self.centroids = torch.randn(m * n, dim, device=device, dtype=torch.float)
        # self.centroids = torch.randn(m * n, dim, device=device, dtype=torch.double)

        locs = [np.array([i, j]) for i in range(self.m) for j in range(self.n)]
        self.locations = torch.LongTensor(np.array(locs)).to(device)

        self.maprange = torch.stack([torch.tensor((self.m, self.n)) for i in range(self.m * self.n)]).float().to(device)

        self.offset1 = torch.tensor([-self.m, -self.n], device=device)
        self.offset2 = torch.tensor([self.m, self.n], device=device)
        self.offset3 = torch.tensor([-self.m, 0], device=device)
        self.offset4 = torch.tensor([self.m, 0], device=device)
        self.offset5 = torch.tensor([0, -self.n], device=device)
        self.offset6 = torch.tensor([0, self.n], device=device)
        self.offset7 = torch.tensor([-self.m, self.n], device=device)
        self.offset8 = torch.tensor([self.m, -self.n], device=device)

        self.precompute = precompute
        if self.precompute:
            # Fast computation is only right for the periodic topology
            if self.periodic:
                self.distance_mat = self.compute_all()
            else:
                self.distance_mat = torch.stack([self.get_bmu_distance_squares(loc) for loc in self.locations])
        self.umat = None

        # Clustering parameters
        self.cluster_att = None
        self.clusters_user = None

    def to_device(self, device):
        for k, v in vars(self).items():
            var = v
            try:
                var = var.to(device)
                self.__dict__[k] = var
                # print(f'{k} -> {device}')
            except AttributeError:
                pass
                # print(f'{k} ... {device}')
        return self

    def get_bmu_distance_squares(self, bmu_loc):
        bmu_loc = bmu_loc.unsqueeze(0).expand_as(self.locations).float()
        if self.periodic:
            d1 = torch.sum(torch.pow(self.locations.float() - bmu_loc, 2), 1)
            d2 = torch.sum(torch.pow(self.locations.float() + self.offset1 - bmu_loc, 2), 1)
            d3 = torch.sum(torch.pow(self.locations.float() + self.offset2 - bmu_loc, 2), 1)
            d4 = torch.sum(torch.pow(self.locations.float() + self.offset3 - bmu_loc, 2), 1)
            d5 = torch.sum(torch.pow(self.locations.float() + self.offset4 - bmu_loc, 2), 1)
            d6 = torch.sum(torch.pow(self.locations.float() + self.offset5 - bmu_loc, 2), 1)
            d7 = torch.sum(torch.pow(self.locations.float() + self.offset6 - bmu_loc, 2), 1)
            d8 = torch.sum(torch.pow(self.locations.float() + self.offset7 - bmu_loc, 2), 1)
            d9 = torch.sum(torch.pow(self.locations.float() + self.offset8 - bmu_loc, 2), 1)
            bmu_distance_squares, _ = torch.min(torch.stack([d1, d2, d3, d4, d5, d6, d7, d8, d9]), 0)
        else:
            bmu_distance_squares = torch.sum(torch.pow(self.locations.float() - bmu_loc, 2), 1)
        return bmu_distance_squares

    def compute_all(self):
        """use the first line computation to get the other ones. Only works for periodic topology"""

        # get first line of the grid this could be sped up with the same trick but indexing might get tricky
        # 'first_line' is thus the m first rows of the distance matrix and has shape m, grid_size
        first_line = [self.get_bmu_distance_squares(loc) for loc in self.locations[:self.m]]
        first_line = torch.cat(first_line)
        first_line = first_line.reshape((self.m, self.grid_size))

        other_lines = []
        for line in range(1, self.n):
            a = first_line[:, (self.grid_size - line * self.m):]
            b = first_line[:, :(self.grid_size - line * self.m)]
            block = torch.cat((a, b), dim=1)
            other_lines.append(block)

        all_lines = torch.cat((first_line, *other_lines), 0)
        return all_lines

    @staticmethod
    def find_batchsize(x):
        """
        Dimension needed is BS, 1(vector), dim
        """
        if len(x.size()) == 1:
            x.unsqueeze_(0)
        batch_size = x.size()[0]
        return x[:, None, :], batch_size

    def scheduler(self, it, tot):
        if self.sched == 'linear':
            return 1.0 - it / tot
        # half the lr 20 times
        if self.sched == 'half':
            return 0.5 ** int(20 * it / tot)
        # decay from 1 to exp(-5)
        if self.sched == 'exp':
            return np.exp(- 5 * it / tot)
        raise NotImplementedError('Wrong value of "sched"')

    def __call__(self, x, learning_rate_op):
        """
        timing info : now most of the time is in pdist ~1e-3s and the rest is 0.2e-3
        :param x: the minibatch
        :param learning_rate_op: the learning rate to apply to the batch
        :return:
        """
        # Make an inference call

        # Compute distances from batch to centroids
        x, batch_size = self.find_batchsize(x)
        dists = torch.cdist(x, self.centroids, p=self.p_norm)

        # Find closest and retrieve the gaussian correlation matrix for each point in the batch
        # bmu_loc is BS, num points
        mindist, bmu_index = torch.min(dists, -1)
        bmu_loc = self.locations[bmu_index].reshape(batch_size, 2)

        # Compute the update

        # Update LR
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op
        self.alpha_op = alpha_op
        self.sigma_op = sigma_op
        if self.precompute:
            bmu_distance_squares = self.distance_mat[bmu_index].reshape(batch_size, self.grid_size)
        else:
            bmu_distance_squares = []
            for loc in bmu_loc:
                bmu_distance_squares.append(self.get_bmu_distance_squares(loc))
            bmu_distance_squares = torch.stack(bmu_distance_squares)
        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, 2 * sigma_op ** 2 + 1e-5)))
        learning_rate_multiplier = alpha_op * neighbourhood_func

        # Take the difference of centroids with centroids and weight it with gaussian
        # x is (BS,1,dim)
        # self.weights is (grid_size,dim)
        # delta is (BS, grid_size, dim)
        expanded_x = x.expand(-1, self.grid_size, -1)
        expanded_weights = self.centroids.unsqueeze(0).expand((batch_size, -1, -1))
        delta = expanded_x - expanded_weights
        delta = torch.mul(learning_rate_multiplier.reshape(*learning_rate_multiplier.size(), 1).expand_as(delta), delta)

        # Perform the update by taking the mean
        delta = torch.mean(delta, dim=0)
        new_weights = torch.add(self.centroids, delta)
        self.centroids = new_weights
        return bmu_loc, torch.mean(mindist)

    def inference_call(self, x):
        """
                timing info : now most of the time is in pdist ~1e-3s and the rest is 0.2e-3
                :param x:
                :param it:
                :return:
                """
        # Compute distances from batch to centroids
        # Dimension needed is BS, 1(vector), dim
        x, batch_size = self.find_batchsize(x)
        dists = torch.cdist(x, self.centroids, p=self.p_norm)

        # Find closest and retrieve the gaussian correlation matrix for each point in the batch
        # bmu_loc is BS, num points
        mindist, bmu_index = torch.min(dists, -1)
        bmu_loc = self.locations[bmu_index].reshape(batch_size, 2)
        return bmu_loc, mindist

    def fit(self, samples, batch_size=20, n_iter=None, print_each=20):
        if self.alpha is None:
            self.alpha = float((self.m * self.n) / samples.shape[0])
        if n_iter is None:
            n_iter = self.niter
        n_steps_periter = len(samples) // batch_size
        total_steps = n_iter * n_steps_periter

        step = 0
        start = time.perf_counter()
        learning_error = list()
        for iter_no in range(n_iter):
            order = np.random.choice(len(samples), size=n_steps_periter, replace=False)
            for counter, index in enumerate(order):
                lr_step = self.scheduler(step, total_steps)
                bmu_loc, error = self.__call__(samples[index:index + batch_size], learning_rate_op=lr_step)
                learning_error.append(error)
                if not step % print_each:
                    print(f'{iter_no + 1}/{n_iter}: {batch_size * (counter + 1)}/{len(samples)} '
                          f'| alpha: {self.alpha_op:4f} | sigma: {self.sigma_op:4f} '
                          f'| error: {error:4f} | time {time.perf_counter() - start:4f}')
                step += 1
        self.compute_umat()
        self.compute_all_dists()
        return learning_error

    def predict(self, samples, batch_size=100):
        """
        Batch the prediction to avoid memory overloading
        """
        batch_size = min(batch_size, len(samples))
        # Avoid empty batches
        n_batch = (len(samples) - 1) // batch_size

        bmus = np.zeros((len(samples), 2))
        errors = list()

        for i in range(n_batch + 1):
            sys.stdout.write(f'{i+1}/{n_batch+1}\r')
            sys.stdout.flush()
            batch = samples[i * batch_size:i * batch_size + batch_size]
            bmu_loc, error = self.inference_call(batch)
            bmus[i * batch_size:i * batch_size + batch_size] = bmu_loc.cpu().numpy()
            errors.append(error)
        errors = torch.cat(errors)
        errors = errors.cpu().numpy()
        return bmus, errors

    @staticmethod
    def _unfold(umat, adj):
        """
        - umat: U-matrix
        - adj: Adjacency matrix of the corresponding U-matrix
        """
        n1, n2 = umat.shape
        mstree = graph.minimum_spanning_tree(adj)
        start = umat.argmin()
        sdist, pred = graph.shortest_path(mstree, indices=start, directed=False,
                                          return_predecessors=True)
        floodpath = np.asarray(np.unravel_index(sdist.argsort(), (n1, n2))).T
        parents_floodpath = np.asarray(np.unravel_index(pred[sdist.argsort()][1:], (n1, n2))).T
        flow = np.sign(floodpath[1:] - parents_floodpath)
        i, j = floodpath[0]
        u, v = 0, 0
        mapping = {(i, j): (u, v)}
        u_min, v_min = 0, 0

        for step, vec in enumerate(flow):
            i, j = floodpath[step + 1]
            u, v = mapping[tuple(parents_floodpath[step])] + vec
            u_min = u if u < u_min else u_min
            v_min = v if v < v_min else v_min
            mapping[(i, j)] = (u, v)
        path = [(u - u_min, v - v_min) for u, v in mapping.values()]
        floodpath = [(i, j) for i, j in floodpath]
        mapping = dict(zip(floodpath, path))
        uumat = np.ones(np.asarray(path).max(axis=0) + 1) * np.inf
        # uumat = np.ones(np.asarray(path).max(axis=0) + 1)
        for k, v in mapping.items():
            uumat[v[0], v[1]] = umat[k[0], k[1]]
        return uumat, mapping

    @staticmethod
    def _get_umat(smap, shape=None, rmsd=False, return_adjacency=False, periodic=True):
        """
        Compute the U-matrix based on a map of centroids and their connectivity.
        """

        def neighbor_dim2_toric(p, s):
            """
            Efficient toric neighborhood function for 2D SOM.
            """
            x, y = p
            X, Y = s
            xm = (x - 1) % X
            ym = (y - 1) % Y
            xp = (x + 1) % X
            yp = (y + 1) % Y
            return [(xm, ym), (xm, y), (xm, yp), (x, ym), (x, yp), (xp, ym), (xp, y), (xp, yp)]

        def neighbor_dim2_grid(p, s):
            """
            Efficient grid neighborhood function for 2D SOM.
            """
            x, y = p
            X, Y = s
            xm = max((x - 1), 0)
            ym = max((y - 1), 0)
            xp = min((x + 1), X - 1)
            yp = min((y + 1), Y - 1)
            shortlist = {(xm, ym), (xm, y), (xm, yp), (x, ym), (x, yp), (xp, ym), (xp, y), (xp, yp)}
            shortlist.discard((x, y))
            return list(shortlist)

        # By default a map is a square
        if shape is None:
            if smap.ndim == 2:
                n = int(np.sqrt(smap.shape[0]))
                smap = smap.reshape((n, n, -1))
            shape = list(smap.shape)[:-1]
        umatrix = np.zeros(shape)
        adjmat = {'data': [], 'row': [], 'col': []}

        for point in itertools.product(*[range(s) for s in shape]):
            neuron = smap[point]
            if periodic:
                neighbors = tuple(np.asarray(neighbor_dim2_toric(point, shape), dtype='int').T)
            else:
                neighbors = tuple(np.asarray(neighbor_dim2_grid(point, shape), dtype='int').T)

            cdist = scipy.spatial.distance.cdist(smap[neighbors], neuron[None], metric='euclidean')
            umatrix[point] = cdist.mean()

            adjmat['row'].extend([np.ravel_multi_index(point, shape), ] * len(neighbors[0]))
            adjmat['col'].extend(np.ravel_multi_index(neighbors, shape))
            adjmat['data'].extend(cdist[:, 0])
        if rmsd:
            natoms = smap.shape[-1] / 3.
            umatrix /= natoms
            umatrix = np.sqrt(umatrix)

        if return_adjacency:
            adjmat = scipy.sparse.coo_matrix((adjmat['data'],
                                              (adjmat['row'], adjmat['col'])),
                                             shape=(np.prod(shape), np.prod(shape)))
            return umatrix, adjmat
        else:
            return umatrix

    def compute_umat(self):
        smap = self.centroids.cpu().numpy().reshape((self.m, self.n, -1))
        umat, adj = self._get_umat(smap, shape=(self.m, self.n), return_adjacency=True, periodic=self.periodic)
        # Renormalize
        umat = (umat - np.min(umat)) / (np.max(umat) - np.min(umat))
        self.umat = umat
        self.adj = adj
        if self.periodic:
            uumat, mapping = self._unfold(umat, adj)
            self.uumat = uumat
            self.mapping = mapping
            self.reversed_mapping = {v: k for k, v in self.mapping.items()}
            umat = uumat
        else:
            self.mapping = {(i, j): (i, j) for (i, j) in itertools.product(range(50), range(50))}
            self.reversed_mapping = {v: k for k, v in self.mapping.items()}
            self.uumat = umat

    def compute_all_dists(self):
        # GRAPH-BASED
        # mstree = graph.minimum_spanning_tree(self.adj)
        adj = self.adj.tocsr()
        self.all_to_all_dist = graph.shortest_path(adj, directed=False)
        # self.all_to_all_dist = graph.shortest_path(mstree, directed=False)

    def cluster(self, min_distance=2):
        """
        Perform clustering based on the umatrix.

        We have tried several methods that can be put into two main categories :
        Graph-based : using either a minimum spanning tree of the full connectivity
        Image-based : using the U-matrix and segmentation techniques. In the periodic case,
                        an unfolding of the umatrix was necessary to do so.
        """

        local_min = peak_local_max(-self.umat, min_distance=min_distance)
        n_local_min = local_min.shape[0]
        clusterer = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=n_local_min)
        labels = clusterer.fit_predict(self.all_to_all_dist)
        labels = labels.reshape((self.m, self.n))

        # IMAGE-BASED
        # from skimage.morphology import disk
        # from skimage.filters import rank
        # from skimage.util import img_as_ubyte
        # from skimage.segmentation import watershed
        # # denoise image
        # umat = img_as_ubyte(umat)
        # denoised = rank.median(umat, disk(2))
        # # find continuous region (low gradient -
        # # where less than 10 for this image) --> markers
        # # disk(5) is used here to get a more smooth image
        # markers = rank.gradient(denoised, disk(5)) < 10
        # markers = ndi.label(markers)[0]
        # # local gradient (disk(2) is used to keep edges thin)
        # gradient = rank.gradient(denoised, disk(2))
        # labels = watershed(gradient, markers)
        # if self.periodic:
        #     square_label = np.zeros((self.m, self.n))
        #     for (u, v), (i, j) in self.reversed_mapping.items():
        #         square_label[i, j] = labels[u, v]
        #     return square_label
        self.cluster_att = labels.flatten()
        return labels

    def manual_cluster(self):
        from .somgui import Wheel, Click

        fig, ax = plt.subplots(figsize=(8, 10))
        cax = ax.matshow(self.umat)
        fig.colorbar(cax)

        click = Click(ax=ax)
        fig.canvas.mpl_connect('button_press_event', click)
        wheel = Wheel(self, click, ax=ax)

        if self.clusters_user is not None:
            wheel.clusters = self.clusters_user
            wheel.expand_clusters()
            wheel.plot_clusters()
        fig.canvas.mpl_connect('scroll_event', wheel)
        fig.canvas.mpl_connect('button_press_event', wheel)
        ax.format_coord = wheel.format_coord
        plt.show()
        self.cluster_att = wheel.expanded_clusters.flatten()
        self.clusters_user = wheel.clusters

    def predict_cluster(self, samples=None, batch_size=100, user=False):
        """
        we have a mapping from each unit to its cluster in the flattened form in self.cluster_att
        Then we need to turn the bmu attributions into the index in this list and return the cluster attributions
        """
        if self.cluster_att is None:
            cluster_att = self.cluster()
            self.cluster_att = cluster_att.flatten()
        if samples is None:
            try:
                self.bmus
            except:
                print('No existing BMUs in the SOM object, one needs data points to predict clusters on')
        else:
            bmus, error = self.predict(samples, batch_size=batch_size)
            self.bmus = bmus
            self.error = error
        flat_bmus = (self.bmus[:, 0] * self.n + self.bmus[:, 1]).astype(np.int32)
        codebook = []
        inds = np.arange(len(self.bmus))
        for i in range(self.m * self.n):
            sel = (flat_bmus == i)
            if sel.sum() > 0:
                ind = inds[sel][self.error[sel].argmin()]
                codebook.append(ind)
            else:
                codebook.append(-1)
        self.codebook = codebook
        if not user:
            return self.cluster_att[flat_bmus], error
        else:
            return self.clusters_user.flatten()[flat_bmus], error


if __name__ == '__main__':
    pass

    # Prepare data
    max_points = 5000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = np.random.rand(10000, 50)
    y = np.random.rand(10000, 1) > 0.5
    X = X[:max_points]
    y = y[:max_points]
    X = torch.from_numpy(X)
    X = X.float()
    X = X.to(device)

    # Create SOM
    n = 3
    somsize = n ** 2
    nsamples = X.shape[0]
    dim = X.shape[1]
    niter = 20
    batch_size = 50
    nsteps = int(nsamples / batch_size)
    som = SOM(n, n, dim, niter=niter, device=device, precompute=True, periodic=True)

    # Fit it and get results
    learning_error = som.fit(X, batch_size=batch_size)
    bmus, inference_error = som.predict(X, batch_size=batch_size)
    predicted_clusts, errors = som.predict_cluster(X[45:56])
    print('some cluster for some random points are : ', predicted_clusts)
