import numpy as np
from PIL import Image
import torch.utils.data as data
import os, random
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


target_transform = Compose([
    RandomCrop(96),
    ToTensor()
])

LR_transform = Compose([
    ToPILImage(),
    Resize(24, interpolation=Image.BICUBIC),
    ToTensor()
])




class TrainData(data.Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        self.data_dir = os.path.join(data_dir, 'train_HR')
        self.target = os.listdir(self.data_dir)

    def __getitem__(self, index):
        target = target_transform(Image.open(os.path.join(self.data_dir, self.target[index])))
        low = LR_transform(target)
       
        return target, low

    def __len__(self):
        return len(self.target)


class testData(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data_dir = './dataset/test'
        self.hr_files = sorted(os.listdir(os.path.join(self.data_dir, 'HR')))
        self.lr_files = sorted(os.listdir(os.path.join(self.data_dir, 'LR')))

    def __getitem__(self, index):
        hr = Image.open(os.path.join(self.data_dir, 'HR',self.hr_files[index]))
        lr = Image.open(os.path.join(self.data_dir, 'LR',self.lr_files[index]))

        return ToTensor()(hr), ToTensor()(lr)
    def __len__(self):
        return len(self.hr_files)
        





        


