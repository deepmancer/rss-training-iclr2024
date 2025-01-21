import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EmbeddingDataset(Dataset):
    def __init__(self, X_data, labels):
        self.embeddings = X_data
        self.labels = np.unique(labels, return_inverse=True)[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding_vector = torch.tensor(self.embeddings[idx])
        label_onehot = torch.tensor(self.labels[idx])
        return embedding_vector, label_onehot


class EmbeddingDataset_ul(Dataset):
    def __init__(self, X_data):
        self.embeddings = X_data

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding_vector = torch.tensor(self.embeddings[idx])
        return embedding_vector


def load_data(config, X_train, y_train, X_val, y_val, X_test, y_test, X_data_ul, has_unlabeled=True):

    dataset = {'train': EmbeddingDataset(X_train, y_train),
              'val': EmbeddingDataset(X_val, y_val),
              'test': EmbeddingDataset(X_test, y_test)}

    dataloaders_labeled = {x: DataLoader(dataset[x], batch_size=config['batch_size_l'], num_workers=1) for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val', 'test']}

    if has_unlabeled:
        dataset_ul = {'train': EmbeddingDataset_ul(X_data_ul)}

        dataloaders_unlabeled = {x: DataLoader(dataset_ul['train'], batch_size=config['batch_size_ul'], num_workers=1) for x in ['train']}

        return y_train, dataloaders_labeled, dataloaders_unlabeled, dataset_sizes
    else:
        return y_train, dataloaders_labeled, None, dataset_sizes