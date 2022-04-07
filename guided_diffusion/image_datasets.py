import os
import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    dataset_mode,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    is_train=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if dataset_mode == 'cityscapes':
        all_files = _list_image_files_recursively(os.path.join(data_dir, 'leftImg8bit', 'train' if is_train else 'val'))
        labels_file = _list_image_files_recursively(os.path.join(data_dir, 'gtFine', 'train' if is_train else 'val'))
        classes = [x for x in labels_file if x.endswith('_labelIds.png')]
        instances = [x for x in labels_file if x.endswith('_instanceIds.png')]
    elif dataset_mode == 'ade20k':
        all_files = _list_image_files_recursively(os.path.join(data_dir, 'images', 'training' if is_train else 'validation'))
        classes = _list_image_files_recursively(os.path.join(data_dir, 'annotations', 'training' if is_train else 'validation'))
        instances = None
    elif dataset_mode == 'celeba':
        # The edge is computed by the instances.
        # However, the edge get from the labels and the instances are the same on CelebA.
        # You can take either as instance input
        all_files = _list_image_files_recursively(os.path.join(data_dir, 'train' if is_train else 'test', 'images'))
        classes = _list_image_files_recursively(os.path.join(data_dir, 'train' if is_train else 'test', 'labels'))
        instances = _list_image_files_recursively(os.path.join(data_dir, 'train' if is_train else 'test', 'labels'))
    else:
        raise NotImplementedError('{} not implemented'.format(dataset_mode))

    print("Len of Dataset:", len(all_files))

    dataset = ImageDataset(
        dataset_mode,
        image_size,
        all_files,
        classes=classes,
        instances=instances,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        is_train=is_train
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_mode,
        resolution,
        image_paths,
        classes=None,
        instances=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        is_train=True
    ):
        super().__init__()
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.local_instances = None if instances is None else instances[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        out_dict = {}
        class_path = self.local_classes[idx]
        with bf.BlobFile(class_path, "rb") as f:
            pil_class = Image.open(f)
            pil_class.load()
        pil_class = pil_class.convert("L")

        if self.local_instances is not None:
            instance_path = self.local_instances[idx] # DEBUG: from classes to instances, may affect CelebA
            with bf.BlobFile(instance_path, "rb") as f:
                pil_instance = Image.open(f)
                pil_instance.load()
            pil_instance = pil_instance.convert("L")
        else:
            pil_instance = None

        if self.dataset_mode == 'cityscapes':
            arr_image, arr_class, arr_instance = resize_arr([pil_image, pil_class, pil_instance], self.resolution)
        else:
            if self.is_train:
                if self.random_crop:
                    arr_image, arr_class, arr_instance = random_crop_arr([pil_image, pil_class, pil_instance], self.resolution)
                else:
                    arr_image, arr_class, arr_instance = center_crop_arr([pil_image, pil_class, pil_instance], self.resolution)
            else:
                arr_image, arr_class, arr_instance = resize_arr([pil_image, pil_class, pil_instance], self.resolution, keep_aspect=False)

        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()
            arr_instance = arr_instance[:, ::-1].copy() if arr_instance is not None else None

        arr_image = arr_image.astype(np.float32) / 127.5 - 1

        out_dict['path'] = path
        out_dict['label_ori'] = arr_class.copy()

        if self.dataset_mode == 'ade20k':
            arr_class = arr_class - 1
            arr_class[arr_class == 255] = 150
        elif self.dataset_mode == 'coco':
            arr_class[arr_class == 255] = 182

        out_dict['label'] = arr_class[None, ]

        if arr_instance is not None:
            out_dict['instance'] = arr_instance[None, ]

        return np.transpose(arr_image, [2, 0, 1]), out_dict


def resize_arr(pil_list, image_size, keep_aspect=True):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if keep_aspect:
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    else:
        pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    return arr_image, arr_class, arr_instance


def center_crop_arr(pil_list, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = (arr_image.shape[0] - image_size) // 2
    crop_x = (arr_image.shape[1] - image_size) // 2
    return arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size],\
           arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size],\
           arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_instance is not None else None


def random_crop_arr(pil_list, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = random.randrange(arr_image.shape[0] - image_size + 1)
    crop_x = random.randrange(arr_image.shape[1] - image_size + 1)
    return arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size],\
           arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size],\
           arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_instance is not None else None
