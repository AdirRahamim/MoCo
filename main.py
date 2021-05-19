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

    parser.add_argument('--lr', type=int, default=0.015,
                        help='lr used for SGD optimizer(pretrain)')
    parser.add_argument('--weight-decay', type=int, default=0.0001,
                        help='Weight decay for SGD optimizer(pretrain)')
    parser.add_argument('--momentum', type=int, default=0.9,
                        help='Momentum for SGD optimizer(pretrain)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size to use')
    parser.add_argument('--epochs', type=int, default=800,
                        help='Num of epochs to train on(pretrain)')

    parser.add_argument('--lr2', type=int, default=0.1,
                        help='lr used for SGD optimizer linear classifier train')
    parser.add_argument('--weight-decay2', type=int, default=0.0,
                        help='Weight decay for SGD optimizer linear classifier train')
    parser.add_argument('--momentum2', type=int, default=0.9,
                        help='Momentum for SGD optimizer linear classifier train')
    parser.add_argument('--batch-size2', type=int, default=64,
                        help='Batch size to use linear classifier train')
    parser.add_argument('--epochs2', type=int, default=200,
                        help='Num of epochs to train on linear classifier train')

    parser.add_argument('--K', type=int, default=65536,
                        help='queue size to use')
    parser.add_argument('--T', type=int, default=0.07,
                        help='Logits temperature')
    parser.add_argument('--m', type=int, default=0.999,
                        help='MoCo momentum value to use to update encoder key parameters')
    parser.add_argument('--feature-dim', type=int, default=128,
                        help='Feature dimension')

    parser.add_argument('--save-path', type=str, default='./checkpoint',
                        help='Path to save checkpoints')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'-> device: {device}')
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

    train_linear_transforms = transforms.Compose([
                                    transforms.RandomResizedCrop(size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(**__imagenet_stats)])

    val_transforms = transforms.Compose([
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(**__imagenet_stats)])

    dataset_train = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'train'), train_transforms)
    dataset_val = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'val'), val_transforms)
    dataset_train_linear = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'train'), train_linear_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=True,
    )

    train_linear_dataloader = torch.utils.data.DataLoader(
        dataset_train_linear,
        batch_size=args.batch_size2,
        num_workers=8,
        drop_last=True,
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size2,
        num_workers=8,
        drop_last=True,
        shuffle=False,
    )

    # Init model
    print('Init model')
    model = MoCo(models.resnet50, args.K, args.T, args.m, args.feature_dim, device).to(device)

    print(f'Unsupervised pretraining started at: {time.ctime()}\n')
    unsupervised_train(train_dataloader, model, device, args)
    print(f'\nUnsupervised pretraining done at: {time.ctime()}\n')

    torch.save(model.state_dict(), os.path.join(args.save_path, f'moco{args.epochs}.pth'))

    print(f'\nLinear classifier training started at: {time.ctime()}\n')
    linear_classifier_train_eval(train_linear_dataloader, val_dataloader, model, device, args)
    print(f'\nLinear classifier training done at: {time.ctime()}\n')

    print('Done!')

def linear_classifier_train_eval(train_loader, val_loader, model, device, args):
    encoder = models.resnet50(num_classes=10).to(device)
    for name, param in encoder.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k.startswith('f_q') and not k.startswith('f_q.fc'):
            state_dict[k[len("f_q."):]] = state_dict[k]
        del state_dict[k]

    encoder.load_state_dict(state_dict, strict=False)

    # Loss function, optimizer and schedualer
    criterion = nn.CrossEntropyLoss().to(device)
    parameters = list(filter(lambda p: p.requires_grad, encoder.parameters()))
    optimizer = torch.optim.SGD(parameters, args.lr2, momentum=args.momentum2, weight_decay=args.weight_decay2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs2, eta_min=0, last_epoch=-1)
    best_acc = 0
    for epoch in range(args.epochs2):
        print(f'Linear: epoch: {epoch}')
        train_linear_for_epoch(train_loader, encoder, criterion, optimizer, device, args)
        scheduler.step()
        acc = validate(val_loader, encoder, criterion, device, args)
        if acc > best_acc:
            best_acc = acc
            torch.save(encoder.state_dict(), os.path.join(args.save_path, f'classifier.pthbest'))

    torch.save(encoder.state_dict(), os.path.join(args.save_path, f'classifier.pth{args.epochs2}'))

def validate(loader, encoder, criterion, device, args):
    encoder.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = encoder(x)
            loss = criterion(outputs, y)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        print(f'Linear [TEST] Acc: {100.*correct/total:.3f}')
        return correct/total


def train_linear_for_epoch(loader, encoder, criterion, optimizer, device, args):
    loader.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = encoder(x)
        loss = criterion(outputs, y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(y)).item()
        total += y.numel()

    print(f'Linear: [TRAIN] loss: {epoch_loss:.3f}')
    print(f'Linear: [TRAIN] Acc: {100.*correct/total:.3f}')

def enqueue_and_dequeue(queue, k):
    return torch.cat([queue, k], dim=0)[k.shape[0]:]

def train_for_epoch(loader, model, criterion, optimizer, queue, device, args):
    model.train()
    epoch_loss = 0
    total = 0
    for (x_q, x_k), _ in tqdm(loader):
        x_q, x_k = x_q.to(device), x_k.to(device)

        logits, labels, k = model(x_q, x_k, queue, return_k=True)
        loss = criterion(logits, labels)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        queue = enqueue_and_dequeue(queue, k)

    return epoch_loss/len(loader), queue

def unsupervised_train(loader, model, device, args):
    # Loss function, optimizer and schedualer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

    # Init queue
    queue = F.normalize(torch.randn(args.K, args.feature_dim).to(device))

    for epoch in range(args.epochs):
        print(f'Feature: epoch: {epoch}')
        epoch_loss, queue = train_for_epoch(loader, model, criterion, optimizer, queue, device, args)
        scheduler.step()
        print(f'Feature: [TRAIN] loss: {epoch_loss:.3f}')

if __name__ == '__main__':
    main()

