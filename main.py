import tarfile
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
import torchvision.models as models
from tqdm import tqdm
import time

from Models.MoCo import MoCo
from utils import TwoCropsTransform


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for MoCoV2 training')
    parser.add_argument('--dataset-filename', type=str, default='imagenette2-160.tgz',
                        help='Dataset to train on')

    parser.add_argument('--lr', type=int, default=0.03,
                        help='lr used for SGD optimizer')
    parser.add_argument('--weight-decay', type=int, default=0.0001,
                        help='Weight decay for SGD optimizer')
    parser.add_argument('--momentum', type=int, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Num of epochs to train on')

    parser.add_argument('--K', type=int, default=65536,
                        help='queue size to use')
    parser.add_argument('--T', type=int, default=0.07,
                        help='Logits temperature')
    parser.add_argument('--m', type=int, default=0.999,
                        help='MoCo momentum value to use to update encoder key parameters')
    parser.add_argument('feature-dim', type=int, default=128,
                        help='Feature dimension')
    args = parser.parse_args()
    return args

def enqueue_and_dequeue(queue, k, batch_size):
    return torch.cat([queue, k], dim=0)[batch_size:]

def train_for_epoch(loader, model, criterion, optimier, queue, args):
    model.train()
    epoch_loss = 0
    for (x_q, x_k), _ in tqdm(loader):
        x_q, x_k = x_q.to(device), x_k.to(device)

        logits, labels, k = model(x_q, x_k, return_k=True)
        loss = criterion(logits, labels)
        epoch_loss += loss

        optimier.zero_grad()
        loss.backward()
        optimier.step()

        queue = enqueue_and_dequeue(queue, k, x_q.shape[0])

    return epoch_loss, queue

if __name__ == '__main__':
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Dataset extraction
    print('Extract dataset and create loaders')
    dataset_filename = args.dataset_filename
    dataset_foldername = dataset_filename.split('.')[0]
    data_path = './data'
    dataset_filepath = dataset_filename
    dataset_folderpath = os.path.join(data_path, dataset_foldername)

    os.makedirs(data_path, exist_ok=True)
    if not os.path.exists(dataset_folderpath):
        with tarfile.open(dataset_filepath, 'r:gz') as tar:
            tar.extractall(path=data_path)

    # Dataloader creation
    size = 224
    ks = (int(0.1 * size) // 2) * 2 + 1
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    train_transforms = TwoCropsTransform(transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=ks)]),  # MoCoV2
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**__imagenet_stats)]))

    val_transforms = transforms.Compose([
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(**__imagenet_stats)])

    dataset_train = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'train'), train_transforms)
    dataset_val = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'val'), val_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=False,
    )

    # Init model
    print('Init model')
    # TODO add arguments
    model = MoCo(models.resnet50, args.K, args.T, args.m, device).to(device)

    # Loss function, optimizer and schedualer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # Init queue
    queue = F.normalize(torch.randn(args.feature_dim, args.K))

    print(f'Training started at: {time.ctime()}')
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        scheduler.step()
        epoch_loss, queue = train_for_epoch(train_dataloader, model, criterion, optimizer, queue, args)
        print(f'[TRAIN] loss: {epoch_loss}')
