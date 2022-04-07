import os
from PIL import Image
from torch.utils import data
from torchvision import transforms


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None, return_path=False, test_list=None):
        if test_list is not None:
            with open(test_list, "r") as f:
                test_files = f.readlines()
            test_files = [x.strip() for x in test_files]
            self.samples = [os.path.join(root, dir) for dir in os.listdir(root) if dir.split('.')[0] in test_files]
        else:
            self.samples = [os.path.join(root, dir) for dir in os.listdir(root)]
        self.samples.sort()
        self.transform = transform
        self.return_path = return_path
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, os.path.basename(fname)
        else:
            return img

    def __len__(self):
        return len(self.samples)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False, return_path=False, test_list=None):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform, return_path=return_path, test_list=test_list)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)