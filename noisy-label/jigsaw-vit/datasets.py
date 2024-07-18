# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_all_classes(*dirs):
    all_classes = set()
    for root_dir in dirs:
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                all_classes.add(class_name)
    return sorted(all_classes)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        nb_cls = 10
    elif args.data_set == 'CIFAR100':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        nb_cls = 100
    elif args.data_set == 'Animal10N':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        nb_cls = 10
    elif args.data_set == 'Clothing1M':
        # we use a randomly selected balanced training subset
        root = os.path.join(args.data_path, 'noisy_rand_subtrain' if is_train else 'clean_val')
        nb_cls = 14
    elif args.data_set == 'Food101N':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        nb_cls = 101
    elif args.data_set == "auto_arborist":
        traindir = os.path.join(args.data_path, 'train')
        valdir = os.path.join(args.data_path, 'val')

        all_classes = get_all_classes(traindir, valdir)

        # Create a universal class_to_idx mapping
        class_to_idx = {class_name: idx for idx,
                        class_name in enumerate(all_classes)}

        root = os.path.join(args.data_path, 'train' if is_train else 'val')
    elif args.data_set == "inat100k":
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        nb_cls = 1000

    dataset = datasets.ImageFolder(root, transform=transform)

    # need to adjust for missing classes for auto arborist
    if args.data_set == "auto_arborist":
        old_classes = dataset.classes

        dataset.class_to_idx = class_to_idx
        dataset.classes = all_classes

        dataset.samples = [
            (path, dataset.class_to_idx[
                    old_classes[target]
                ]) for path, target in dataset.samples
        ]
        dataset.targets = [s[1] for s in dataset.samples]

        nb_cls = len(dataset.classes)

    print(dataset)

    return dataset, nb_cls


def build_transform(is_train, args):
    if args.data_set == 'CIFAR10' or args.data_set == 'CIFAR100':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.data_set == "inat100k":
        mean = (0.4839, 0.4956, 0.4215)
        std = (0.2170, 0.2131, 0.2460)
    else:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
