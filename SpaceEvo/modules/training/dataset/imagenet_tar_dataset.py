import numpy as np
import os
import tarfile
import torchvision.transforms as transforms

from torch.utils import data
from PIL import Image


class ImageTarDataset(data.Dataset):
    def __init__(self, tar_file, transform=transforms.ToTensor(), return_labels=True):
        '''
        return_labels:
        Whether to return labels with the samples
        transform:
        A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        '''
        self.tar_file = tar_file
        self.tar_handle = None
        categories_set = set()
        self.tar_members = []
        self.categories = {}
        self.categories_to_examples = {}
        with tarfile.open(tar_file, 'r:') as tar:
            for index, tar_member in enumerate(tar.getmembers()):
                if tar_member.name.count('/') != 2:
                    continue
                category = self._get_category_from_filename(tar_member.name)
                categories_set.add(category)
                self.tar_members.append(tar_member)
                cte = self.categories_to_examples.get(category, [])
                cte.append(index)
                self.categories_to_examples[category] = cte
        categories_set = sorted(categories_set)
        for index, category in enumerate(categories_set):
            self.categories[category] = index
        self.num_examples = len(self.tar_members)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        print("Loaded the dataset from {}. It contains {} samples.".format(
            tar_file, self.num))
        self.return_labels = return_labels
        self.transform = transform

    def _get_category_from_filename(self, filename):
        begin = filename.find('/')
        begin += 1
        end = filename.find('/', begin)
        return filename[begin:end]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        index = self.indices[index]
        if self.tar_handle is None:
            self.tar_handle = tarfile.open(self.tar_file, 'r:')

        sample = self.tar_handle.extractfile(self.tar_members[index])
        image = Image.open(sample).convert('RGB')
        image = self.transform(image)

        if self.return_labels:
            category = self.categories[self._get_category_from_filename(
                self.tar_members[index].name)]
            return image, category
        return image


if __name__ == '__main__':

    # path to the imagenet validation dataset, both options would work
    imagenet_valid = os.path.join(os.environ['AMLT_DATA_DIR'], 'imagenet_valid.tar')

    # load dataset
    print('Loading dataset with various transforms')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = ImageTarDataset(
        imagenet_valid, return_labels=True, transform=transform)
    print('Building loader')
    loader = data.DataLoader(dataset)
    for x in loader:
        # print first image
        print(x)
        break
