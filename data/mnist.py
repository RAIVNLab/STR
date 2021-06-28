from typing import Any, Callable, Optional, Tuple
from torchvision import datasets, transforms
from PIL import Image
from args import args
import os
import torch
import torchvision
import numpy as np


class MultiMNIST:
    def __init__(self, args):
        super(MultiMNIST, self).__init__()

        data_root = os.path.join(args.data, "mnist")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            MultiMNISTDataset(
                data_root,
                train=True,
                download=True,
                num_concat=args.num_concat,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            MultiMNISTDataset(
                data_root,
                num_concat=args.num_concat,
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )


class MultiMNISTDataset(datasets.MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download : bool = False,
        num_concat : int = 1,
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.length = int(super().__len__() ** num_concat)
        if self.train:
            self.length = args.num_train_examples or self.length
        else:
            self.length = args.num_val_examples or self.length

        self.num_concat = num_concat

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Pick 4 random
        if self.train:
            rng = np.random.RandomState(index*2)
        else:
            rng = np.random.RandomState(index*2 + 1)

        
        indices = rng.randint(0, super().__len__(), (self.num_concat,))
        img, target = self.data[indices], self.targets[indices]
        base = 10 ** torch.arange(self.num_concat - 1, -1, -1)

        img = torch.cat([img[i] for i in range(self.num_concat)], dim=-1)
        target = (base * target).sum()

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
