import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
import wandb


def load_data(batch_size: int, fold: int = -1, resize: bool = True, crop: bool = False):
    print("Loading CIFAR-10 dataset...")
    augmentation_transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(10),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
    if resize:
        augmentation_transforms = [torchvision.transforms.Resize((224, 224))] + augmentation_transforms
    if crop:
        augmentation_transforms = augmentation_transforms[:-1] + [torchvision.transforms.RandomResizedCrop((32, 32), (0.8, 1))] + augmentation_transforms[-1:]
    augmentation_transforms = torchvision.transforms.Compose(augmentation_transforms)

    transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]
    if resize:
        transforms = [torchvision.transforms.Resize((224, 224))] + transforms
    transforms = torchvision.transforms.Compose(transforms)

    train_iter = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=augmentation_transforms)
    val_iter = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)

    indices = list(range(len(train_iter)))
    if fold > -1:
        kfold = KFold(n_splits=5, random_state=2023, shuffle=True)
        split = list(kfold.split(indices))[fold]
        train_idxs, val_idxs = split[0], split[1]
    else:
        np.random.shuffle(indices)
        train_idxs, val_idxs = indices[:45000], indices[-5000:]

    train_dataloader = DataLoader(train_iter, batch_size=batch_size, sampler=SubsetRandomSampler(train_idxs))
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, sampler=SubsetRandomSampler(val_idxs))

    test_iter = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
    test_dataloader = DataLoader(test_iter, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def train_epoch(train_dataloader: DataLoader, model: nn.Module, optimizer: Optimizer,
                scheduler: LRScheduler = None, device = torch.device('cpu'), epoch = None, logging: bool = False):
    model.train()
    losses = 0

    for i, (images, labels) in enumerate(tqdm(train_dataloader)):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses += loss.item()
        
        if (((i + 1) % 4 == 0) or (i + 1 == len(train_dataloader))) and logging:
            wandb.log({ "loss": loss.item(), "lr": optimizer.param_groups[0]['lr'] })
        
        if scheduler:
            scheduler.step(epoch + i / len(train_dataloader))

    return losses / len(train_dataloader)


def evaluate(val_dataloader: DataLoader, model: nn.Module, device = torch.device('cpu')):
    model.eval()
    losses = 0

    for images, labels in tqdm(val_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        losses += loss.item()

    return losses / len(val_dataloader)

def score(test_dataloader: DataLoader, model: nn.Module, device = torch.device('cpu')):
    model.eval()
    losses = 0
    acc = 0

    for i, (images, labels) in enumerate(tqdm(test_dataloader)):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        losses += loss.item()
        _, max = torch.max(logits, dim=-1)
        acc += torch.sum(max == labels).item()
        
    return losses / len(test_dataloader), acc / 10000