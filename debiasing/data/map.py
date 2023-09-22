from torch.utils.data import Dataset


class MapDataset(Dataset):
    """Given a dataset, creates a dataset which applies a mapping function to its items (lazily, only when an item is called).
    Note that data is not cloned/copied from the initial dataset.
    
    Args:
        dataset: dataset
        map_fn: lambda
    """
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn
    
    def __getitem__(self, index):
        return self.map(*self.dataset[index])
    
    def __len__(self):
        return len(self.dataset)