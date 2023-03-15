from itertools import chain

import h5py
import numpy as np
import torch
from mani_skill2.utils.io_utils import load_json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class DemoDataset(Dataset):

    def __init__(self, dataset_file, load_count=-1):
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.observations = []
        self.actions = []
        self.total_frames = 0
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            # we use :-1 here to ignore the last observation as that 
            # is the terminal observation which has no actions
            self.observations.append(trajectory["obs"][:-1])
            self.actions.append(trajectory["actions"])
        self.observations = np.vstack(self.observations)
        self.actions = np.vstack(self.actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = torch.from_numpy(self.actions[idx]).float()
        obs = torch.from_numpy(self.observations[idx]).float()

        return obs, action

    @staticmethod
    def collate_fn(samples):
        obs = torch.stack([sample[0] for sample in samples])
        act = torch.stack([sample[1] for sample in samples])

        return obs, act

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle,
            num_workers=num_workers, collate_fn=DemoDataset.collate_fn)


class DemoNpzDataset(Dataset):

    def __init__(self, demo_path):
        demos = np.load(demo_path)
        self.observations = demos["observations"]
        self.actions = demos["actions"]

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = torch.from_numpy(self.actions[idx]).float()
        obs = torch.from_numpy(self.observations[idx]).float()

        return obs, action

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle,
            num_workers=num_workers)
