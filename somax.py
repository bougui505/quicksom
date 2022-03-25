#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import datetime
from functools import partial
import itertools
import matplotlib.pyplot as plt
import os
import pickle
import sys
import time

import numpy as np
import scipy.spatial
import scipy.sparse

import scipy.sparse.csgraph as graph
from skimage.feature import peak_local_max
from sklearn.cluster import AgglomerativeClustering

import torch

import jax
import jax.numpy as jnp


def distance_one(x, y):
    return jnp.linalg.norm(x - y)


jax_cdist_0 = jax.vmap(distance_one, (None, 0))
jax_cdist = jax.vmap(jax_cdist_0, (0, None))
jax_cdist_0 = jax.jit(jax_cdist_0)
jax_cdist = jax.jit(jax_cdist)


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, nparr):
        self.nparr = nparr

    def __len__(self):
        return len(self.nparr)

    def __getitem__(self, idx):
        return idx, self.nparr[idx]


def build_dataloader(dataset, num_workers, batch_size, shuffle=True):
    if not isinstance(dataset, torch.utils.data.Dataset) and not isinstance(dataset, torch.utils.data.DataLoader):
        dataset = ArrayDataset(dataset)
    if isinstance(dataset, torch.utils.data.Dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=num_workers, pin_memory=True)
    if isinstance(dataset, torch.utils.data.DataLoader):
        dataloader = dataset
    dataloader.collate_fn = jax_collate

    return dataloader


def jax_collate(items):
    """

    :param items: items is a list of tuples (id, vector)
    :return:
    """
    labels = [item[0] for item in items]
    np_data = [item[1] for item in items]
    np_data = np.stack(np_data)
    return labels, np_data


class SOM:
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
    :param precompute: Speedup for initialization. Creates a little overhead for very small trainings
    :param periodic: Boolean to use a periodic topology
    :param metric: takes as input two arrays (n,p) and (m,p) and returns a distance matrix (n,m)
    :param centroids: Initial som map to use. No random initialization
     """

    def __init__(self,
                 m,
                 n,
                 dim,
                 alpha=None,
                 sigma=None,
                 n_epoch=2,
                 sched='linear',
                 device=jax.devices()[0],
                 precompute=True,
                 periodic=False,
                 metric=None,
                 centroids=None):

        # topology of the som
        super(SOM, self).__init__()
        self.device = device
        self.step = 0
        self.m = m
        self.n = n
        self.grid_size = m * n
        self.dim = dim
        self.periodic = periodic
        if metric is None:
            self.metric = jax_cdist
        else:
            self.metric = metric

        # optimization parameters
        self.sched = sched
        self.n_epoch = n_epoch
        if alpha is not None:
            self.alpha = float(alpha)
        else:
            self.alpha = alpha
        if sigma is None:
            self.sigma = np.sqrt(self.m * self.n) / 2.0
        else:
            self.sigma = float(sigma)

        if centroids is None:
            self.centroids = jax.device_put(np.random.randn(m * n, dim), device=device)
        else:
            self.centroids = centroids

        locs = np.stack([np.array([i, j]) for i in range(self.m) for j in range(self.n)])
        self.locations = jax.device_put(locs, device=device)
        self.precompute = precompute
        if self.precompute:
            # Fast computation is only right for the periodic topology
            if self.periodic:
                self.distance_mat = self.compute_all()
            else:
                self.distance_mat = jnp.stack([self.get_bmu_distance_squares(loc) for loc in self.locations])
            self.distance_mat = jax.device_put(self.distance_mat, device=device)
        self.umat = None

        # Clustering parameters
        self.cluster_att = None
        self.clusters_user = None

    def to_device(self, device):
        """
        Send the jax objects to device
        :param device:
        :return:
        """
        for k, v in vars(self).items():
            var = v
            if isinstance(var, jax.interpreters.xla._DeviceArray):
                var = jax.device_put(var, device=device)
                self.__dict__[k] = var
        return self

    # @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def get_bmu_distance_squares(self, bmu_loc):
        if self.periodic:
            offsets = jnp.array([[0, 0],
                                 [-self.m, -self.n],
                                 [self.m, self.n],
                                 [-self.m, 0],
                                 [self.m, 0],
                                 [0, -self.n],
                                 [0, self.n],
                                 [-self.m, self.n],
                                 [self.m, -self.n]])
            # offsets = jax.device_put(offsets)
            offset_locations = jnp.array([self.locations + offset for offset in offsets])
            batch_cdist = jax.vmap(jax_cdist_0, (None, 0))
            distances = batch_cdist(bmu_loc, offset_locations)
            min_dists = jnp.min(distances, axis=0)
            return min_dists
        else:
            distances = jax_cdist_0(bmu_loc, self.locations)
            return distances

    def compute_all(self):
        """
        Use the first line computation to get the other ones. Only works for periodic topology
        """

        # get first line of the grid this could be sped up with the same trick but indexing might get tricky
        # 'first_line' is thus the m first rows of the distance matrix and has shape m, grid_size
        import time
        first_line = [self.get_bmu_distance_squares(loc) for loc in self.locations[:self.m]]
        first_line = jnp.stack(first_line)
        # We use square euclidean distance.
        first_line = first_line ** 2
        other_lines = []
        for line in range(1, self.n):
            first_part = first_line[:, (self.grid_size - line * self.m):]
            second_part = first_line[:, :(self.grid_size - line * self.m)]
            block = jnp.concatenate((first_part, second_part), axis=1)
            other_lines.append(block)
        all_lines = jnp.concatenate((first_line, *other_lines), 0)
        return all_lines

    @staticmethod
    def find_batchsize(x):
        """
        Dimension needed is BS, 1(vector), dim
        """
        if len(x.shape) == 1:
            return x[None, None, :], 1
        batch_size = x.shape[0]
        return x[:, None, :], batch_size

    def scheduler(self, it, tot):
        if self.sched == 'linear':
            return 1.0 - it / tot
        # half the lr 20 times
        if self.sched == 'half':
            return 0.5 ** int(20 * it / tot)
        # decay from 1 to exp(-5)
        if self.sched == 'exp':
            return np.exp(-5 * it / tot)
        raise NotImplementedError('Wrong value of "sched"')

    def __call__(self, x, learning_rate_op):
        """
        timing info : now most of the time is in pdist ~1e-3s and the rest is 0.2e-3
        :param x: the minibatch
        :param learning_rate_op: the learning rate to apply to the batch
        :return:
        """
        bmu_loc, error, self.centroids = self.static_call(x, learning_rate_op, self.centroids)
        return bmu_loc, error

    @partial(jax.jit, static_argnums=(0,))
    def static_call(self, x, learning_rate_op, self_centroids):
        """
        timing info : now most of the time is in pdist ~1e-3s and the rest is 0.2e-3
        :param x: the minibatch
        :param learning_rate_op: the learning rate to apply to the batch
        :param self_centroids: to be compiled, the centroids cannot be fed through self since they will be muted.
        :return:
        """
        # Make an inference call
        # Compute distances from batch to centroids
        x, batch_size = self.find_batchsize(x)
        dists = self.metric(x, self_centroids)

        # x is of shape (30,1,50), centroids of shape (100, 50)

        # Find closest and retrieve the gaussian correlation matrix for each point in the batch
        # bmu_loc is BS, num points
        bmu_index = jnp.argmin(dists, -1)
        mindist = jnp.take_along_axis(dists, bmu_index[:, None], axis=-1)
        bmu_loc = self.locations[bmu_index].reshape(batch_size, 2)
        # Compute the update

        # Update LR
        # It is a matrix of shape (BS, centroids) and tell for each input how much it will affect each centroid
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op
        if self.precompute:
            bmu_distance_squares = self.distance_mat[bmu_index].reshape(batch_size, self.grid_size)
        else:
            bmu_distance_squares = []
            for loc in bmu_loc:
                bmu_distance_squares.append(self.get_bmu_distance_squares(loc))
            bmu_distance_squares = jnp.stack(bmu_distance_squares)
        neighbourhood_func = jnp.exp(-(bmu_distance_squares / (2 * sigma_op ** 2 + 1e-5)))
        learning_rate_multiplier = alpha_op * neighbourhood_func

        # Take the difference of centroids with input and weight it with gaussian
        # x is (BS,1,dim)
        # self.weights is (grid_size,dim)
        # delta is (BS, grid_size, dim)
        def flat_form(one_x, alpha, weight):
            small_d = one_x - weight
            mutliplied = small_d * alpha[:, None]
            return mutliplied

        vdelta = jax.vmap(flat_form, in_axes=(0, 0, None))
        delta = vdelta(x, learning_rate_multiplier, self_centroids)
        # Perform the update by taking the mean
        delta = jnp.mean(delta, axis=0)
        new_weights = self_centroids + delta
        return bmu_loc, jnp.mean(mindist), new_weights

    # @partial(jax.jit, static_argnums=(0,))
    def inference_call_one(self, x):
        """
        timing info : now most of the time is in pdist ~1e-3s and the rest is 0.2e-3
        :param x:
        :param it:
        :return:
        """
        # Compute distances from batch to centroids
        # Dimension needed is BS, 1(vector), dim
        x, batch_size = self.find_batchsize(x)
        dists = self.metric(x, self.centroids)

        # Find closest and retrieve the gaussian correlation matrix for each point in the batch
        bmu_index = jnp.argmin(dists, -1)
        mindist = jnp.take_along_axis(dists, bmu_index[:, None], axis=-1)
        return bmu_index, mindist

    # @partial(jax.jit, static_argnums=(0, 2))
    def inference_call_two(self, x, n_bmu=2):
        """
        timing info : now most of the time is in pdist ~1e-3s and the rest is 0.2e-3
        :param x:
        :param it:
        :return:
        """
        # Compute distances from batch to centroids
        x, batch_size = self.find_batchsize(x)
        dists = self.metric(x, self.centroids)
        idx = np.argsort(dists, -1)
        selected = idx[:, :1]
        mindists = jnp.take_along_axis(dists, selected, axis=-1)
        return selected, mindists

    def inference_call(self, x, n_bmu=1):
        """
        timing info : now most of the time is in pdist ~1e-3s and the rest is 0.2e-3
        :param x:
        :param it:
        :return:
        """
        # Find closest and retrieve the gaussian correlation matrix for each point in the batch
        if n_bmu == 1:
            return self.inference_call_one(x)
        else:
            # In that case, return the bmu indices.
            return self.inference_call_two(x, n_bmu)

    def loc_from_idx(self, idx):
        loc = self.locations[idx]
        loc = jnp.reshape(loc, (-1, 2))
        return loc

    def fit(self,
            dataset=None,
            batch_size=20,
            n_epoch=None,
            print_each=100,
            do_compute_all_dists=True,
            unfold=True,
            normalize_umat=True,
            sigma=None,
            alpha=None,
            logfile='som.log',
            num_workers=os.cpu_count()):
        """
        samples: torch tensor with all the data. If given dataloader must not be given
        dataset: torch data loader object. If given samples must not be given
        """
        if logfile is not None:
            logfile = open(logfile, 'w', buffering=1)
            logfile.write('#epoch #iter #alpha #sigma #error #runtime\n')
        dataloader = build_dataloader(dataset, num_workers, batch_size=batch_size)
        nbatch = len(dataloader)

        if self.alpha is None:
            self.alpha = float((self.m * self.n) / nbatch)
            print('alpha:', self.alpha)
        if sigma is not None:
            # reset the sigma
            self.sigma = sigma
        if alpha is not None:
            # reset the alpha
            self.alpha = alpha
        if n_epoch is None:
            n_epoch = self.n_epoch
        npts = len(dataloader.dataset)
        total_steps = npts * n_epoch
        start = time.perf_counter()
        learning_error = list()
        for epoch in range(n_epoch):
            for i, (label, batch) in enumerate(dataloader):
                lr_step = self.scheduler(self.step, total_steps)
                batch = jax.device_put(batch, self.device)
                bmu_loc, error, new_centroids = self.static_call(batch,
                                                                 learning_rate_op=lr_step,
                                                                 self_centroids=self.centroids)
                self.centroids = new_centroids
                learning_error.append(error)
                if not i % print_each:
                    runtime = time.perf_counter() - start
                    eta = total_steps * runtime / (self.step + batch_size) - runtime
                    print(f'{epoch + 1}/{n_epoch}: {self.step}/{total_steps} '
                          f'| alpha: {self.alpha * lr_step:4f} | sigma: {self.sigma * lr_step:4f} '
                          f'| error: {error:4f} | time {runtime:4f}'
                          f'| eta: {str(datetime.timedelta(seconds=eta))}',
                          flush=True)
                    if logfile is not None:
                        logfile.write(
                            f'{epoch} {self.step} {self.alpha * lr_step} {self.sigma * lr_step} {error} {runtime}\n')
                self.step += batch_size
        self.compute_umat(unfold=unfold, normalize=normalize_umat)
        if do_compute_all_dists:
            self.compute_all_dists()
        if logfile is not None:
            logfile.close()
        return learning_error

    def predict(self, dataset, batch_size=100, print_each=100,
                return_density=False, return_errors=False, num_workers=os.cpu_count()):
        """
        Batch the prediction to avoid memory overloading
        """
        dataloader = build_dataloader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
        bmus = list()
        errors = list()
        if return_density:
            density = np.zeros((self.m, self.n))
        if return_errors:
            bmu_indices = []
        labels = []
        start = time.perf_counter()
        for i, (label, batch) in enumerate(dataloader):
            batch = jax.device_put(batch, self.device)

            # If we want the topographic error, we need to compute more neighbors, and keep the indices
            bmu_idx, error = self.inference_call(batch, n_bmu=2 if return_errors else 1)
            if return_errors:
                bmu_indices.append(bmu_idx)

            # Then we can keep the first only and compute its bmu affectation
            first_idx = bmu_idx[:, 0] if return_errors else bmu_idx
            bmu_loc = self.loc_from_idx(first_idx)
            labels.extend(label)
            bmus.append(bmu_loc)
            if return_density:
                density[tuple(np.asarray(bmu_loc).T)] += 1.
            if error.ndim == 0:
                error = error[None, ...]
            errors.append(error)
        bmus = jnp.concatenate(bmus)
        bmus = np.asarray(bmus)
        errors = jnp.concatenate(errors)
        errors = np.asarray(errors).flatten()

        default_return = [bmus, errors, labels]
        if return_density:
            density /= density.sum()
            default_return.append(density)
        # Optionnally compute errors
        if return_errors:
            quantization_error = np.mean(errors[:, :, 0])
            topo_dists = np.array([self.distance_mat[int(first), int(second)] for first, second in bmu_indices])
            topo_error = np.sum(topo_dists > 1) / len(topo_dists)
            print(f'On these samples, the quantization error is {quantization_error:1f} '
                  f'and the topological error rate is {topo_error:1f}')
            default_return.append(quantization_error)
            default_return.append(topo_error)
        return tuple(default_return)

    def plot_component_plane(self, plane, savefig=None, show=True):
        """
        Get component plane plot : The value of each dimension for each centroid
        """
        smap = self.centroids.cpu().numpy().reshape((self.m, self.n, -1))
        assert 0 <= plane < self.dim
        component_plane = smap[:, :, plane]

        plt.matshow(component_plane)
        plt.colorbar()
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()
        plt.clf()

    @staticmethod
    def _unfold(umat, adj):
        """
        - umat: U-matrix
        - adj: Adjacency matrix of the corresponding U-matrix
        """
        n1, n2 = umat.shape
        mstree = graph.minimum_spanning_tree(adj)
        start = umat.argmin()
        sdist, pred = graph.shortest_path(mstree, indices=start, directed=False, return_predecessors=True)
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

    def _get_umat(self, smap, shape=None, rmsd=False, return_adjacency=False, periodic=True):
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

        start = time.perf_counter()
        for i, point in enumerate(itertools.product(*[range(s) for s in shape])):
            # if not i % 100:
            #     print(f'{i} | time: {str(datetime.timedelta(seconds=time.perf_counter() - start))} ')
            neuron = smap[point][None, :]
            if periodic:
                neighbors = tuple(np.asarray(neighbor_dim2_toric(point, shape), dtype='int').T)
            else:
                neighbors = tuple(np.asarray(neighbor_dim2_grid(point, shape), dtype='int').T)
            smap_neighbor = smap[neighbors]
            cdists = np.asarray(self.metric(smap_neighbor, neuron))
            umatrix[point] = cdists.mean()

            adjmat['row'].extend([np.ravel_multi_index(point, shape), ] * len(neighbors[0]))
            adjmat['col'].extend(np.ravel_multi_index(neighbors, shape))
            adjmat['data'].extend(cdists[:, 0])
        if rmsd:
            natoms = smap.shape[-1] / 3.
            umatrix /= natoms
            umatrix = np.sqrt(umatrix)

        if return_adjacency:
            adjmat = scipy.sparse.coo_matrix((adjmat['data'], (adjmat['row'], adjmat['col'])),
                                             shape=(np.prod(shape), np.prod(shape)))
            return umatrix, adjmat
        else:
            return umatrix

    def compute_umat(self, unfold=True, normalize=True):
        smap = np.asarray(self.centroids).reshape((self.m, self.n, -1))
        umat, adj = self._get_umat(smap=smap, shape=(self.m, self.n),
                                   return_adjacency=True, periodic=self.periodic)
        if normalize:
            # Renormalize
            umat = (umat - np.min(umat)) / (np.max(umat) - np.min(umat))
        self.umat = umat
        self.adj = adj
        if self.periodic and unfold:
            uumat, mapping = self._unfold(umat, adj)
            self.uumat = uumat
            self.mapping = mapping
            self.reversed_mapping = {v: k for k, v in self.mapping.items()}
            umat = uumat
        else:
            self.mapping = {(i, j): (i, j) for (i, j) in itertools.product(range(self.m), range(self.n))}
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
        try:
            labels = clusterer.fit_predict(self.all_to_all_dist)
        except ValueError as e:
            print(f'WARNING : The following error was catched : "{e}"\n'
                  f'The clusterer yields zero clusters on the data.'
                  ' You should train it more or gather more data')
            labels = np.zeros(self.m * self.n)
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

    def manual_cluster(self, autocluster=False):
        # from .somgui import Wheel, Click

        fig, ax = plt.subplots(figsize=(8, 10))
        cax = ax.matshow(self.umat)
        fig.colorbar(cax)
        plt.show()

        click = Click(ax=ax)
        fig.canvas.mpl_connect('button_press_event', click)
        wheel = Wheel(self, click, ax=ax)

        if autocluster:
            self.clusters_user = self.cluster() + 1
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
            bmus, error, labels = self.predict(samples, batch_size=batch_size)
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

    def save_pickle(self, outname):
        del self.device
        del self.metric
        pickle.dump(self, open(outname, 'wb'))

    @staticmethod
    def load_pickle(inname, metric=None, device=jax.devices()[0]):
        loaded_som = pickle.load(open(inname, 'rb'))
        if metric is None:
            print('WARNING : loading a JAX SOM object without its metric results in the cdist metric being used '
                  'because jitted functions can not be pickled. If you want to use another metric, please '
                  'specify it in the SOM.load_pickle function')
            metric = jax_cdist
        loaded_som.device = device
        loaded_som.metric = metric

        # We need to manually choose which arrays are to be put, otherwise every numpy item ends up a jax object.
        loaded_som.centroids = jax.device_put(loaded_som.centroids, device=device)
        loaded_som.locations = jax.device_put(loaded_som.locations, device=device)
        loaded_som.distance_mat = jax.device_put(loaded_som.distance_mat, device=device)
        return loaded_som


def time_som(som, X):
    som.alpha = 0.5
    import time

    X = jax.device_put(X)
    a = time.perf_counter()
    for _ in range(1000):
        som(X[:30], 1)
    jax.block_until_ready(som.centroids)
    print('total time : ', time.perf_counter() - a)

    som.alpha = 0.5
    import time

    a = time.perf_counter()
    # First call is compilation
    _, _, new_centroids = som.static_call(X[:30], 1, som.centroids)
    som.centroids = new_centroids
    jax.block_until_ready(som.centroids)
    print('compilation time : ', time.perf_counter() - a)

    # Second call is runtime
    a = time.perf_counter()
    for _ in range(1000):
        _, _, new_centroids = som.static_call(X[:30], 1, som.centroids)
        som.centroids = new_centroids
    jax.block_until_ready(som.centroids)
    print('total time : ', time.perf_counter() - a)
    sys.exit()


if __name__ == '__main__':
    pass
    # Prepare data
    np.random.seed(0)
    X = np.random.rand(1000, 500)

    # Create SOM
    n = 10
    somsize = n ** 2
    nsamples = X.shape[0]
    dim = X.shape[1]
    niter = 5
    batch_size = 30
    nsteps = int(nsamples / batch_size)
    som = SOM(n, n, dim, n_epoch=niter, precompute=True, periodic=True)
    som.to_device(jax.devices()[0])

    # Fit it and get results
    learning_error = som.fit(X, batch_size=batch_size)
    bmus, inference_error, labels = som.predict(X, batch_size=batch_size)
    predicted_clusts, errors = som.predict_cluster(X[45:56])
    print('some cluster for some random points are : ', predicted_clusts)
