import torch
import gdown
import tarfile
import os

from glob import glob
from torch.utils.data.dataset import Dataset
from PIL import Image

class BFFHQ(Dataset):
    def __init__(self, root, split, percent, transform=None, image_path_list=None):
        super().__init__()

        if not os.path.isdir(os.path.join(root, 'bffhq')):
            self.download_dataset(root)
        root = os.path.join(root, 'bffhq', percent)

        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))

        elif split=='test':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))
            data_conflict = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
                if target_label != bias_label:
                    data_conflict.append(path)
            self.data = data_conflict
    
    def download_dataset(self, path):
        url = "https://drive.google.com/file/d/1hRmeIxhoa6YsyAm2LUPqa7nY1KtOpp6G/view?usp=sharing"
        output = os.path.join(path, 'bffhq.tar.gz')
        print(f'=> Downloading BFFHQ dataset from {url}')
        gdown.download(url, output, quiet=False, fuzzy=True)

        print('=> Extracting dataset..')
        tar = tarfile.open(os.path.join(path, 'bffhq.tar.gz'), 'r:gz')
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
    dataset = BFFHQ(f'/data', 'train', "0.5pct")

    train_target_attr = []
    for data in dataset.data:
        train_target_attr.append(int(data.split('_')[-2]))
    train_target_attr = torch.LongTensor(train_target_attr)

    print(len(dataset))
    print(train_target_attr.max() + 1)