import os
import pathlib
from typing import List, Dict
import numpy as np
import torch
from torch_geometric.data.data import BaseData
import torch_geometric.utils
from torch_geometric.data import Dataset

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

def sample_data_with_given_indices(data, indices):
    return [data[i] for i in indices]

from pathlib import Path

def to_tensor(data_list):
    return [torch.FloatTensor(adj) for adj in data_list]

class DiverseGraphsDataset(Dataset):
    def __init__(self, 
                 dataset_name,
                 graphs_file,
                 split, #[train/test/val]
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 ):
        
        
        self.graphs_file = graphs_file
        self.dataset_name = dataset_name
        self.split = split
        
        
        self.split_to_length = {}

        
        
        super().__init__(root=root, transform=transform, pre_filter=pre_filter,
                         pre_transform=pre_transform)
        
        self.split_processed_dir = Path(self.processed_dir) / self.split    
        
        
        
        
    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return [self.split + '.pt']
        
    def len(self) -> int:
        
        return len(os.listdir(os.path.join(self.processed_dir, self.split)))

    
    def download(self):
        """
        Replaced download interface so that we process oour graphs locally
        """
        
        graphs_file: List[np.ndarray] or Dict[str, np.ndarray]= np.load(self.graphs_file, 
                                                allow_pickle=True,
                                                ) # type: ignore
        
        
        if isinstance(graphs_file, np.lib.npyio.NpzFile):
            assert all((
                "train" in graphs_file,
                "valid" in graphs_file,
                "test" in graphs_file,
            ))
                        
            train_data = to_tensor(graphs_file["train"])
            val_data = to_tensor(graphs_file["valid"])
            test_data = to_tensor(graphs_file["test"])
            
            
            train_len = len(train_data)
            val_len = len(val_data)
            test_len = len(test_data)
            
            
            
        elif isinstance(graphs_file, (np.ndarray, list)):
            
            adjacencies: List[torch.FloatTensor] = to_tensor(graphs_file)
            
            
            num_of_graphs = len(adjacencies)
            
            generator_cpu = torch.Generator()
            generator_cpu.manual_seed(0)
            
            test_len = int(num_of_graphs * 0.05)
            train_len = int((num_of_graphs - test_len) * 0.90)
            val_len = num_of_graphs - train_len - test_len
            indices = torch.randperm(n=num_of_graphs, generator=generator_cpu)
            
            
            train_indices = indices[:train_len]
            val_indices = indices[train_len:train_len+val_len]
            test_indices = indices[train_len + val_len:]
            
            train_data = sample_data_with_given_indices(data=adjacencies, indices=train_indices)
            val_data = sample_data_with_given_indices(data=adjacencies, indices=val_indices)
            test_data = sample_data_with_given_indices(data=adjacencies, indices=test_indices)
            
        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])
        
        self.split_to_length = dict(zip(["train", "val", "test"], [train_len, val_len, test_len]))
            
        print(f"Dataset length are: {train_len=} {val_len=} {test_len=}")
        print("Completed preprocessing of graphs")
    
    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])
        
        data_all = []
        
        self.split_processed_dir = Path(self.processed_dir) / self.split
        self.split_processed_dir.mkdir(exist_ok=True, parents=True)
        
        for i, adj in enumerate(raw_dataset):
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float() # NOTE what is y?
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_all.append(data)
            filename = self.split_processed_dir / f"data_{i}.pt"
            
            torch.save(data, filename)
        
        torch.save(data_all, os.path.join(self.processed_dir, f"{self.split}.pt"))

    def get(self, idx: int):
        
        filename = self.split_processed_dir / f"data_{idx}.pt"
        
        data = torch.load(filename)
        
        return data
        
        
                    

class DiverseGraphsDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=100):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {'train': DiverseGraphsDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path, graphs_file=self.cfg.dataset.graphs_file,
                                                 
                                                 ),
                    'val': DiverseGraphsDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path, graphs_file=self.cfg.dataset.graphs_file
                                        ),
                    'test': DiverseGraphsDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path, graphs_file=self.cfg.dataset.graphs_file
                                        )}
        
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]
    

class DiverseDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'diverse_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
