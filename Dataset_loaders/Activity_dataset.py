
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Self, Callable, Tuple
import torch
from scipy import sparse
from functools import wraps

class ActivityDataset:
    def __init__(self):
        self.train = None
        self.test = None
        self.val = None
        self.col_names = []

    class CustomDataset(Dataset):
        def __init__(self, x, y) -> None:
            self.data_x = torch.tensor(x, dtype=torch.float32)
            self.data_y = torch.tensor(y, dtype=torch.float32)

        def __getitem__(self, index):
            x = self.data_x[index]
            y = self.data_y[index]
            return x, y

        def __len__(self) -> float:
            return len(self.data_x)

    def __load_numpy__(self):
        with np.load('./Dataset/dataset_activity.npz', allow_pickle=True) as dataset:
            self.col_names = dataset['col_names'].tolist()
            train_x = dataset['train_x']
            train_y = dataset['train_y']
            val_x = dataset['val_x']
            val_y = dataset['val_y']
            test_x = dataset['test_x']
            test_y = dataset['test_y']
        return train_x, train_y, val_x, val_y, test_x, test_y


    def load(self, num_of_workers=16) -> Self:
        train_x, train_y, val_x, val_y, test_x, test_y = self.__load_numpy__()

        train = self.CustomDataset(train_x, train_y)
        self.train = DataLoader(train, batch_size=1024, shuffle=True, num_workers=num_of_workers)

        val = self.CustomDataset(val_x, val_y)
        self.val = DataLoader(val, batch_size=1024, num_workers=num_of_workers, pin_memory=True)

        test = self.CustomDataset(test_x, test_y)
        self.test = DataLoader(test, batch_size=1024, num_workers=num_of_workers)
        return self

    def get_dataloaders(self):
        return self.train, self.val, self.test

