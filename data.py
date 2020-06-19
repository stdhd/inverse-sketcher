import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import torchvision.datasets


class SketchyVectors(torch.utils.data.Dataset):

    def __init__(self, split='train'):
        super().__init__()


    def __getitem__(self, idx):
        '''Implements the ``[idx]`` method. Here we convert the numpy data to
        torch tensors.
        '''

        #return sketch_condition, photo

    def __len__(self):
        return len(self.labels)

