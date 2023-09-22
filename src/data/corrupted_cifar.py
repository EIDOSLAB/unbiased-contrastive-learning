import gdown
import tarfile
import os

from glob import glob
from torch.utils.data.dataset import Dataset
from PIL import Image

class CorruptedCIFAR10(Dataset):
    def __init__(self, root, split, percent, transform=None, image_path_list=None):
        super().__init__()
        
        if not os.path.isdir(os.path.join(root, 'cifar10c')):
            self.download_dataset(root)
        root = os.path.join(root, 'cifar10c', percent)

        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split == 'train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split == 'valid':
            self.data = glob(os.path.join(root,split,"*", "*"))

        elif split == 'test':
            self.data = glob(os.path.join(root, '../test',"*","*"))

    def download_dataset(self, path):
        url = "https://drive.google.com/file/d/1_eSQ33m2-okaMWfubO7b8hhvLMlYNJP-/view?usp=sharing"
        output = os.path.join(path, 'cifar10c.tar.gz')
        print(f'=> Downloading corrupted CIFAR10 dataset from {url}')
        gdown.download(url, output, quiet=False, fuzzy=True)

        print('=> Extracting dataset..')
        tar = tarfile.open(os.path.join(path, 'cifar10c.tar.gz'), 'r:gz')
        tar.extractall(path=path)
        tar.close()
        os.remove(output)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, bias = int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label, bias

if __name__ == '__main__':
    dataset = CorruptedCIFAR10(f'/home/{os.environ.get("USER")}/temp', 'train', "0.5pct")