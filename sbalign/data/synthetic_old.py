import numpy as np

import torch
import torch.distributions as td
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import sbalign.utils.helper as helper


def build_data(problem_name, batch_size, device):
    if helper.is_toy_dataset(problem_name):
        return {
            "gmm": MixMultiVariateNormal,
            "checkerboard": CheckerBoard,
            "spiral": Spiral,
            "moon": Moon,
        }.get(problem_name)(batch_size)

    elif helper.is_shape_dataset(problem_name):
        dataset_generator = {}.get(problem_name)
        dataset = dataset_generator()
        return DataSampler(dataset, batch_size, device)

    else:
        raise RuntimeError()


def transform_data(data):
    pass


def random_transformation(data):
    return data


class MixMultiVariateNormal:
    def __init__(self, batch_size, radius=12, num=8, sigmas=None):
        # build mu's and sigma's
        arc = 2 * np.pi / num
        xs = [np.cos(arc * idx) * radius for idx in range(num)]
        ys = [np.sin(arc * idx) * radius for idx in range(num)]
        mus = [torch.Tensor([x, y]) for x, y in zip(xs, ys)]
        dim = len(mus[0])
        sigmas = [torch.eye(dim) for _ in range(num)] if sigmas is None else sigmas

        if batch_size % num != 0:
            raise ValueError("Batch size must be devided by number of Gaussian")
        self.num = num
        self.batch_size = batch_size
        self.dists = [
            td.multivariate_normal.MultivariateNormal(mu, sigma)
            for mu, sigma in zip(mus, sigmas)
        ]

    def log_prob(self, x):
        # assume equally-weighted
        densities = [torch.exp(dist.log_prob(x)) for dist in self.dists]
        return torch.log(sum(densities) / len(self.dists))

    def sample(self):
        ind_sample = self.batch_size / self.num
        samples = [dist.sample([int(ind_sample)]) for dist in self.dists]
        samples = torch.cat(samples, dim=0)
        return samples


class CheckerBoard:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        n = self.batch_size
        n_points = 3 * n
        n_classes = 2
        freq = 5
        x = np.random.uniform(
            -(freq // 2) * np.pi, (freq // 2) * np.pi, size=(n_points, n_classes)
        )
        mask = np.logical_or(
            np.logical_and(np.sin(x[:, 0]) > 0.0, np.sin(x[:, 1]) > 0.0),
            np.logical_and(np.sin(x[:, 0]) < 0.0, np.sin(x[:, 1]) < 0.0),
        )
        y = np.eye(n_classes)[1 * mask]
        x0 = x[:, 0] * y[:, 0]
        x1 = x[:, 1] * y[:, 0]
        sample = np.concatenate([x0[..., None], x1[..., None]], axis=-1)
        sqr = np.sum(np.square(sample), axis=-1)
        idxs = np.where(sqr == 0)
        samples = np.delete(sample, idxs, axis=0)
        # res=res+np.random.randn(*res.shape)*1
        samples = samples[0:n, :]

        # transform dataset by adding constant shift
        samples_t = samples + np.array([0, 3])

        return {"initial": torch.Tensor(samples_t),
                "final": torch.Tensor(samples)}


class Spiral:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        n = self.batch_size
        theta = (
            np.sqrt(np.random.rand(n)) * 3 * np.pi - 0.5 * np.pi
        )

        r_a = theta + np.pi
        data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
        x_a = data_a + 0.25 * np.random.randn(n, 2)
        samples = np.append(x_a, np.zeros((n, 1)), axis=1)
        samples = samples[:, 0:2]

        # rotate and shrink samples
        samples_t = np.zeros(samples.shape)
        for i, v in enumerate(samples):
            samples_t[i] = rotate2d(v, 30)
        samples = samples * 0.8

        return {"initial": torch.Tensor(samples_t),
                "final": torch.Tensor(samples)}


class Moon:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        n = self.batch_size
        x = np.linspace(0, np.pi, n // 2)
        u = np.stack([np.cos(x) + 0.5, -np.sin(x) + 0.2], axis=1) * 10.0
        u += 0.5 * np.random.normal(size=u.shape)
        v = np.stack([np.cos(x) - 0.5, np.sin(x) - 0.2], axis=1) * 10.0
        v += 0.5 * np.random.normal(size=v.shape)
        samples = np.concatenate([u, v], axis=0)

        # rotate and shrink samples
        samples_t = np.zeros(samples.shape)
        for i, v in enumerate(samples):
            samples_t[i] = rotate2d(v, 180)
        # samples = samples * 0.8

        return {"initial": torch.Tensor(samples_t),
                "final": torch.Tensor(samples)}


class DataSampler:  # a dump data sampler
    def __init__(self, dataset, batch_size, device):
        self.num_sample = len(dataset)
        self.dataloader = setup_loader(dataset, batch_size)
        self.batch_size = batch_size
        self.device = device

    def sample(self):
        data = next(self.dataloader)
        return data[0].to(self.device)


def setup_loader(dataset, batch_size):
    train_loader = DataLoaderX(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    print("number of samples: {}".format(len(dataset)))

    while True:
        yield from train_loader


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def rotate2d(x, radians):
    """Build a rotation matrix in 2D, take the dot product, and rotate."""
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, x)

    return m
