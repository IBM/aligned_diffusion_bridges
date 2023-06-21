import torch
from torch_geometric.data import Dataset


class ListDataset(Dataset):

    def __init__(self, processed_dir, id_list):
        super().__init__()

        self.full_processed_dir = processed_dir
        self.id_list = id_list

    def len(self):
        return len(self.id_list)

    def get(self, idx):
        conf_pair_id = self.id_list[idx]

        conf_pair_out = torch.load(f"{self.full_processed_dir}/{conf_pair_id}.pt")
        conf_pair_out.conf_id = conf_pair_id
        return conf_pair_out.clone()
