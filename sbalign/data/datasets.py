import os

import torch
import numpy as np
import torch.distributions as td
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import pathlib

import sbalign.utils.helper as helper
from sbalign.utils.sb_utils import sample_from_brownian_bridge
from sbalign.training.diffusivity import get_diffusivity_schedule


def build_data(problem_name, n_samples, device):
    if helper.is_toy_dataset(problem_name):
        return {
            "gmm": MixMultiVariateNormal,
            "checkerboard": CheckerBoard,
            "spiral": Spiral,
            "moon": Moon,
            "matching_with_exception": MatchingWithException,
            "diagonal_exception": DiagonalMatching
        }.get(problem_name)(n_samples)

    elif helper.is_shape_dataset(problem_name):
        dataset_generator = {}.get(problem_name)
        dataset = dataset_generator()
        return DataSampler(dataset, n_samples, device)

    else:
        raise RuntimeError()


def build_data_loader(args):
    if args.transform is None:
        transform = BrownianBridgeTransform(g=get_diffusivity_schedule(args.diffusivity_schedule, args.max_diffusivity))

    if helper.is_toy_dataset(args.dataset):
        train_dataset = SyntheticDataset(root=os.path.join("../reproducibility/", args.data_dir), transform=transform, problem=args.dataset,
                                         n_samples=args.n_samples, mode="train", split_fracs=args.split_fracs)
        val_dataset = SyntheticDataset(root=os.path.join("../reproducibility/", args.data_dir), transform=transform, problem=args.dataset,
                                        n_samples=args.n_samples, mode="val", split_fracs=None)
    elif helper.is_cell_dataset(args.dataset):
        train_dataset = SavedDataset(args.dataset, transform=transform, n_samples=args.n_samples, mode="train", root_dir="../reproducibility/")
        val_dataset = SavedDataset(args.dataset, transform=transform, n_samples=args.n_samples, mode="val", root_dir="../reproducibility/")

    elif helper.is_shape_dataset(args.dataset):
        pass
    
    elif helper.is_protein_dataset(args.dataset):
        pass

    else:
        raise ValueError(f"{args.dataset} is not supported")

    train_loader = train_dataset.create_loader(batch_size=args.train_bs, 
                                               num_workers=args.num_workers, 
                                               shuffle=True)
    val_loader = val_dataset.create_loader(batch_size=args.val_bs,
                                           num_workers=args.num_workers,
                                           shuffle=True)
    return train_loader, val_loader


class BrownianBridgeTransform(BaseTransform):

    def __init__(self, g):
        self.g = g

    def __call__(self, data):
        bs = data.pos_0.shape[0]
        t = torch.rand((bs, 1))
        return self.apply_transform(data, t)

    def apply_transform(self, data, t):
        # assert (data.pos_0[:,1] == data.pos_T[:,1]).all(), (data.pos_0[:,1], data.pos_T[:,1])
        data.pos_t = sample_from_brownian_bridge(g=self.g, t=t, x_0=data.pos_0, x_T=data.pos_T, t_min=0.0, t_max=1.0)
        data.t = t
        return data


class SavedDataset(Dataset):

    def __init__(self, dataset_name, transform=None, n_samples: int=10000, mode="train", root_dir=pathlib.Path(__file__).parent.resolve(), device='cpu'):
        self.dataset_name = dataset_name
        self.transform = transform
        self.mode = mode

        self.n_samples = n_samples
        self.mode = mode

        dataset_root_path = os.path.join(root_dir, dataset_name)

        self.data = {
            time: np.load(os.path.join(dataset_root_path, "data", self.get_partition_name(time)))[:n_samples] for time in ['initial', 'final']
        }

        # Convert into torch tensors
        self.data = {
            time: torch.from_numpy(self.data[time]).to(device) for time in self.data
        }

    def get_partition_name(self, time):
        return f"{self.dataset_name}_embs_{time}_{self.mode}.npy"

    def __len__(self):
        return self.data["initial"].shape[0]

    def __getitem__(self, index):
        return (self.data["initial"][index], self.data["final"][index])

    def collate_fn(self, data_batch):
        data = Data()
        pos_0, pos_T = zip(*data_batch)

        data.pos_0 = torch.stack(pos_0, dim=0)
        data.pos_T = torch.stack(pos_T, dim=0)
        assert data.pos_0.shape == data.pos_T.shape

        if self.transform is not None:
            return self.transform(data)
        return data

    def create_loader(self,
                      batch_size: int,
                      num_workers: int,
                      shuffle: bool = False):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn)



class SyntheticDataset(Dataset):

    def __init__(self, root: str, problem: str = 'moon', transform=None, 
                 n_samples: int = 10000, mode = "train", split_fracs=None, device='cpu'):
        self.root = root
        self.transform = transform
        self.problem = problem

        self.n_samples = n_samples
        self.mode = mode

        filename = f"{root}/synthetic_data/{problem}_nsamples={n_samples}.pt"
        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            data_dict = self.generate_and_split_data(n_samples, split_fracs)
            torch.save(data_dict, filename)

        data_full = torch.load(filename, map_location=device)
        self.data = data_full[mode]

    def __len__(self):
        return len(self.data["initial"])

    def __getitem__(self, index):
        return (self.data["initial"][index], self.data["final"][index])

    def generate_and_split_data(self, n_samples, split_fracs):
        generator_cls = {
            "gmm": MixMultiVariateNormal,
            "checkerboard": CheckerBoard,
            "spiral": Spiral,
            "moon": Moon,
            "matching_with_exception": MatchingWithException,
            "diagonal_matching": DiagonalMatching,
            "diagonal_matching_inverse": DiagonalMatchingInverse,
        }.get(self.problem)

        generator_fn = generator_cls(n_samples=n_samples)
        data_dict = generator_fn.sample()

        if type(split_fracs[0]) is int:
            assert np.sum(split_fracs) == n_samples
            n_train, n_val, _ = split_fracs

        else:
            n_train = int(split_fracs[0] * n_samples)
            n_val = int(split_fracs[1] * n_samples)

        idxs_permuted = torch.randperm(n_samples)
        idxs_train = idxs_permuted[:n_train]
        idxs_val = idxs_permuted[n_train: n_train + n_val]
        idxs_test = idxs_permuted[n_train + n_val:]

        data_train = {
            'initial': data_dict['initial'][idxs_train],
            'final': data_dict['final'][idxs_train]
        }

        data_val = {
            'initial': data_dict['initial'][idxs_val],
            'final': data_dict['final'][idxs_val]
        }

        data_test = {
            'initial': data_dict['initial'][idxs_test],
            'final': data_dict['final'][idxs_test]
        }

        train_mean, train_std = {}, {}
        
        for key in ['initial', 'final']:
            train_mean_key = torch.mean(data_train[key], dim=0, keepdim=True)
            train_std_key = torch.std(data_train[key], dim=0, keepdim=True)
            
            train_mean[key] = train_mean_key
            train_std[key] = train_std_key

        # for key in ['initial', 'final']:
        #     data_val_key = (data_val[key] - train_mean[key]) / train_std[key]
        #     data_test_key = (data_test[key] - train_mean[key]) / train_std[key]

        #     data_val[key] = data_val_key
        #     data_test[key] = data_test_key

        return dict(train=data_train, val=data_val, test=data_test, 
                    train_mean=train_mean, train_std=train_std)

    def collate_fn(self, data_batch):
        data = Data()
        pos_0, pos_T = zip(*data_batch)

        data.pos_0 = torch.stack(pos_0, dim=0)
        data.pos_T = torch.stack(pos_T, dim=0)
        assert data.pos_0.shape == data.pos_T.shape

        if self.transform is not None:
            return self.transform(data)
        return data

    def create_loader(self,
                      batch_size: int,
                      num_workers: int,
                      shuffle: bool = False):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn)


class MixMultiVariateNormal:

    def __init__(self, n_samples, radius=12, num=8, sigmas=None):
        # build mu's and sigma's
        arc = 2 * np.pi / num
        xs = [np.cos(arc * idx) * radius for idx in range(num)]
        ys = [np.sin(arc * idx) * radius for idx in range(num)]
        mus = [torch.Tensor([x, y]) for x, y in zip(xs, ys)]
        dim = len(mus[0])
        sigmas = [torch.eye(dim) for _ in range(num)] if sigmas is None else sigmas

        if n_samples % num != 0:
            raise ValueError("Batch size must be devided by number of Gaussian")
        self.num = num
        self.n_samples = n_samples
        self.dists = [
            td.multivariate_normal.MultivariateNormal(mu, sigma)
            for mu, sigma in zip(mus, sigmas)
        ]

    def log_prob(self, x):
        # assume equally-weighted
        densities = [torch.exp(dist.log_prob(x)) for dist in self.dists]
        return torch.log(sum(densities) / len(self.dists))

    def sample(self):
        ind_sample = self.n_samples / self.num
        samples = [dist.sample([int(ind_sample)]) for dist in self.dists]
        samples = torch.cat(samples, dim=0)
        return samples


class CheckerBoard:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample(self):
        n = self.n_samples
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

        return {"initial": torch.Tensor(samples_t).float(),
                "final": torch.Tensor(samples).float()}


class Spiral:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample(self):
        n = self.n_samples
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

        return {"initial": torch.Tensor(samples_t).float(),
                "final": torch.Tensor(samples).float()}


class Moon:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample(self):
        n = self.n_samples
        x = np.linspace(0, np.pi, n // 2)
        u = np.stack([np.cos(x) + 0.5, -np.sin(x) + 0.2], axis=1) * 10.0
        u += 0.5 * np.random.normal(size=u.shape)
        u /= 3
        v = np.stack([np.cos(x) - 0.5, np.sin(x) - 0.2], axis=1) * 10.0
        v += 0.5 * np.random.normal(size=v.shape)
        v /= 3
        samples = np.concatenate([u, v], axis=0)

        # rotate and shrink samples
        samples_t = np.zeros(samples.shape)
        for i, v in enumerate(samples):
            samples_t[i] = rotate2d(v, 180)
        # samples = samples * 0.8

        return {"initial": torch.Tensor(samples_t).float(),
                "final": torch.Tensor(samples).float()}

class MatchingWithException:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample(self):
        n = self.n_samples

        left_cloud = np.stack([np.random.normal(-5, .25, size=(n,)), np.linspace(-3.5, 3.5, n)], axis=1)
        right_cloud = np.stack([np.random.normal(5, .25, size=(n,)), np.linspace(-3.5, 3.5, n)], axis=1)

        exceptions_num = np.maximum((25*n)//100, 1)
        print(exceptions_num)

        left_cloud = np.roll(left_cloud, exceptions_num, axis=0)

        # TODO: Check that SyntheticDataset shuffles all samples before extracting train/val slices 
        rand_shuffling = np.random.permutation(left_cloud.shape[0])
        left_cloud = left_cloud[rand_shuffling]
        right_cloud = right_cloud[rand_shuffling]

        return {
            "initial": torch.from_numpy(left_cloud).float(),
            "final": torch.from_numpy(right_cloud).float()
        }


class DiagonalMatching:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample(self):
        n = self.n_samples//2+1

        left_square = np.stack([np.random.uniform(-1.2, -1., size=(n,))*2-5, np.linspace(-.1, .5, n)*4+3], axis=1)
        right_square = np.stack([np.random.uniform(1., 1.2, size=(n,))*2+5, np.linspace(-.1, .5, n)*4+3], axis=1)

        top_square = np.stack([np.linspace(-.3, .3, n)*4, np.random.uniform(.8, 1., size=(n,))*2+3], axis=1)
        bottom_square = np.stack([np.linspace(-.3, .3, n)*4, np.random.uniform(-1.5, -1.3, size=(n,))*2-3], axis=1)

        rand_shuffling = np.random.permutation(self.n_samples)

        return {
            "initial": torch.from_numpy(np.concatenate([left_square, top_square], axis=0)[:self.n_samples][rand_shuffling]).float(),
            "final": torch.from_numpy(np.concatenate([right_square, bottom_square], axis=0)[:self.n_samples][rand_shuffling]).float()
        }


class DiagonalMatchingInverse:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample(self):
        n = self.n_samples//2+1

        left_square = np.stack([np.random.uniform(-1.2, -1., size=(n,))*2-5, np.linspace(-.1, .5, n)*4+3], axis=1)
        right_square = np.stack([np.random.uniform(1., 1.2, size=(n,))*2+5, np.linspace(-.1, .5, n)*4+3], axis=1)

        top_square = np.stack([np.linspace(-.3, .3, n)*4, np.random.uniform(.8, 1., size=(n,))*2+3], axis=1)
        bottom_square = np.stack([np.linspace(-.3, .3, n)*4, np.random.uniform(-1.5, -1.3, size=(n,))*2-3], axis=1)

        rand_shuffling = np.random.permutation(self.n_samples)

        return {
            "initial": torch.from_numpy(np.concatenate([right_square, bottom_square], axis=0)[:self.n_samples][rand_shuffling]).float(),
            "final": torch.from_numpy(np.concatenate([left_square, top_square], axis=0)[:self.n_samples][rand_shuffling]).float(),
        }


class DataSampler:  # a dump data sampler

    def __init__(self, dataset, n_samples, device):
        self.num_sample = len(dataset)
        self.dataloader = setup_loader(dataset, n_samples)
        self.n_samples = n_samples
        self.device = device

    def sample(self):
        data = next(self.dataloader)
        return data.to(self.device)


def setup_loader(dataset, n_samples):
    train_loader = DataLoaderX(
        dataset, n_samples=n_samples, shuffle=True, num_workers=0, drop_last=True
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
