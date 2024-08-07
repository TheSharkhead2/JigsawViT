# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class CustomImageFolder(ImageFolder):
    def __init__(
                 self,
                 root,
                 transform=None,
                 target_transform=None,
                 class_to_idx=None):
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())

            # removes any samples that weren't in train
            self.samples = [(path, self.class_to_idx[target])
                            for path, target in self.samples
                            if target in self.class_to_idx]

            # recompute targets
            self.targets = [self.class_to_idx[s[1]] for s in self.samples]


def build_dataset(is_train, args, class_to_idx, all_classes):
    transform = build_transform(is_train, args)

    if args.data_set in ["auto_arborist", "inat100"]:
        # if eval, take test directory
        if args.eval:
            if args.data_set == "inat100":
                root = os.path.join(args.data_path, "val_new")
            else:
                root = os.path.join(args.data_path, "test")

            dataset = datasets.ImageFolder(root, transform=transform)

            nb_classes = len(dataset.classes)
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)

            old_classes = dataset.classes

            dataset.class_to_idx = class_to_idx
            dataset.classes = all_classes

            # need to reset samples and targets
            dataset.samples = [
                (path, dataset.class_to_idx[
                        old_classes[target]
                    ]) for path, target in dataset.samples]
            dataset.targets = [s[1] for s in dataset.samples]

            nb_classes = len(dataset.classes)
    else:
        if args.data_set == 'CIFAR':
            dataset = datasets.CIFAR100(args.data_path,
                                        train=is_train, transform=transform)
            nb_classes = 100
        elif args.data_set == 'IMNET':
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
            nb_classes = 1000
        elif args.data_set == 'INAT':
            dataset = INatDataset(args.data_path, train=is_train, year=2018,
                                  category=args.inat_category,
                                  transform=transform)
            nb_classes = dataset.nb_classes
        elif args.data_set == 'INAT19':
            dataset = INatDataset(args.data_path, train=is_train, year=2019,
                                  category=args.inat_category,
                                  transform=transform)
            nb_classes = dataset.nb_classes

    # elif args.data_set == 'auto_arborist':
    #     # validation data in test dir
    #     root = os.path.join(args.data_path, 'train' if is_train else 'val')
    #     dataset = datasets.ImageFolder(root, transform=transform)

    #     # if not training, make sure to load with correct class mappings
    #     if not is_train:
    #         dataset.class_to_idx = class_to_idx
    #         dataset.classes = list(class_to_idx.keys())

    #     nb_classes = len(dataset.classes)

    # elif args.data_set == "inat100":
    #     root = os.path.join(args.data_path, 'train' if is_train else 'val')
    #     dataset = datasets.ImageFolder(root, transform=transform)

    #     if not is_train:
    #         dataset.class_to_idx = class_to_idx
    #         dataset.classes = list(class_to_idx.keys())

    #     nb_classes = len(dataset.classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
