# 定义数据集
from PIL import Image
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils import data
from sklearn.model_selection import train_test_split

class ComDataset(Dataset):
    def __init__(self, clean_path, list_path, transform,mode):
        self.path = clean_path  # 图像路径
        self.transform = transform  #数据转换
        self.list_path = list_path
        self.test_dataset = []
        self.mode = mode
        self.preprocess()
        self.num_images = len(self.test_dataset)

    def __len__(self):
        return self.num_images

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.list_path, 'r')]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            self.test_dataset.append(filename)

        print('Finished preprocessing the dataset...')


    def __getitem__(self, index):
        filename = self.test_dataset[index]
        filepath = self.path + filename   # 图片文件的路径

        # 将图片转变为数据
        with Image.open(filepath) as f:
            img = f.convert('RGB')
            img = self.transform(img)
        
        return img, filename


def get_loader(clean_path, list_path, image_size=256, 
               batch_size=1,num_workers=1,shuffle=False,mode="test"):
    """Build and return a data loader."""
    transform = []
    # if mode == 'train':
    #     transform.append(T.RandomHorizontalFlip())
    transform.append(T.Resize([image_size,image_size]))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    dataset = ComDataset(clean_path, list_path,transform,mode)
    data_loader = data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return data_loader