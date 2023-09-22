import datetime
import math
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import argparse
import data
import models
import losses
import time
import wandb
import torch.utils.tensorboard

from torchvision import transforms
from torchvision import datasets
from util import AverageMeter, NViewTransform, accuracy, ensure_dir, set_seed, arg2bool, warmup_learning_rate
from lars import LARS

def parse_arguments():
    parser = argparse.ArgumentParser(description="Contrastive debiasing",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
    parser.add_argument('--log_dir', type=str, help='tensorboard log dir', default='logs')

    parser.add_argument('--data_dir', type=str, help='path of data dir', required=True, default='/data')
    parser.add_argument('--dataset', type=str, help='dataset (format name_attr e.g. biased-mnist_0.999)', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)

    parser.add_argument('--epochs', type=int, help='number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--warm', type=arg2bool, help='warmup lr', default=False)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step', 'none'], default='cosine')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="100,150")
    parser.add_argument('--optimizer', type=str, help="optimizer (adam, sgd or lars)", choices=["adam", "sgd", "lars"], default="sgd")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-4)

    parser.add_argument('--model', type=str, help='model architecture')

    parser.add_argument('--method', type=str, help='loss function', choices=['infonce', 'infonce-strong'], default='infonce')
    parser.add_argument('--n_views', type=int, help='number of different views', default=1)
    parser.add_argument('--selfsup', type=arg2bool, help='do not use labels', default=False)
    parser.add_argument('--augplus', type=arg2bool, help='use simclr aug (for selfsup)', default=False)

    parser.add_argument('--form', type=str, help='loss form (in or out)', default='out')
    parser.add_argument('--temp', type=float, help='supcon/infonce temperature', default=0.1)
    parser.add_argument('--epsilon', type=float, help='infonce epsilon', default=0.)
    parser.add_argument('--lr_epsilon', type=float, help='epsilon lr', default=1e-4)
    
    parser.add_argument('--lambd', type=float, help='lagrangian weight for debiasing', default=0.)
    parser.add_argument('--lambd_alpha_ratio', type=float, help='compute lambd as alpha*ratio (0=disabled)', default=0)
    parser.add_argument('--kld', type=float, help='weight of std term', default=1.)

    parser.add_argument('--alpha', type=float, help='infonce weight', default=1.)
    parser.add_argument('--alpha_rand', action='store_true', help='sample alpha randomly in [0,1]')

    parser.add_argument('--beta', type=float, help='cross-entropy weight WITH supcon', default=0)
    
    parser.add_argument('--feat_dim', type=int, help='size of projection head', default=128)
    parser.add_argument('--mlp_lr', type=float, help='mlp lr', default=0.001)
    parser.add_argument('--mlp_lr_decay', type=str, help='mlp lr decay', default='constant')
    parser.add_argument('--mlp_max_iter', type=int, help='mlp training epochs', default=500)
    parser.add_argument('--mlp_optimizer', type=str, help='mlp optimizer', default='adam')
    parser.add_argument('--mlp_batch_size', type=int, help='mlp batch size', default=None)
    parser.add_argument('--test_freq', type=int, help='test frequency', default=1)
    parser.add_argument('--train_on_head', type=arg2bool, help="train clf on projection head features", default=True)

    parser.add_argument('--amp', action='store_true', help='use amp')

    opts = parser.parse_args()
    if opts.alpha_rand:
        opts.alpha = random.random()
        print("Sampling random alpha", opts.alpha)

    if opts.lambd_alpha_ratio > 0:
        opts.lambd = opts.lambd_alpha_ratio * opts.alpha
        print("lambda/alpha ratio ->", opts.lambd_alpha_ratio, "*", opts.alpha, "=", opts.lambd) 

    if opts.selfsup and opts.n_views == 1:
        print("n_views must be > 1 if selfsup is true")
        exit(1)

    return opts

def load_data(opts):
    if opts.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    elif opts.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    elif 'biased-mnist' in opts.dataset:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    elif 'corrupted-cifar10' in opts.dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    elif 'imagenet' in opts.dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    
    elif 'bffhq' in opts.dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    if 'cifar' in opts.dataset:
        T_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        T_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    elif 'biased-mnist' in opts.dataset:
        T_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        T_test = T_train

    elif opts.dataset in ['imagenet100', 'imagenet']:
        resize_size = 256
        crop_size = 224
        T_train = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if opts.augplus:
            T_train = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
                    
        T_test = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if opts.augplus:
            T_test = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    
    elif 'bffhq' in opts.dataset:
        T_train = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        T_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    if hasattr(opts, 'n_views'):
        T_train = NViewTransform(T_train, opts.n_views)

    if opts.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opts.data_dir, transform=T_train, train=True, download=True)
        train_dataset = data.MapDataset(train_dataset, lambda x, y: (x, y, 0))

        test_dataset = datasets.CIFAR10(root=opts.data_dir, transform=T_test, train=False, download=True)
        test_dataset = data.MapDataset(test_dataset, lambda x, y: (x, y, 0))

        opts.n_classes = 10

    elif opts.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opts.data_dir, transform=T_train, train=True, download=True)
        train_dataset = data.MapDataset(train_dataset, lambda x, y: (x, y, 0))

        test_dataset = datasets.CIFAR100(root=opts.data_dir, transform=T_test, train=False, download=True)
        test_dataset = data.MapDataset(test_dataset, lambda x, y: (x, y, 0))
        opts.n_classes = 100

    elif 'corrupted-cifar10' in opts.dataset:
        corruption = opts.dataset.replace('corrupted-cifar10_', '')
        opts.corruption = corruption

        train_dataset = data.CorruptedCIFAR10(root=opts.data_dir, split="train", percent=corruption, transform=T_train)
        test_dataset = data.CorruptedCIFAR10(root=opts.data_dir, split="test", percent=corruption, transform=T_test)
        opts.n_classes = 10

    elif 'bffhq' in opts.dataset:
        print("Loading BFFHQ")
        percent = opts.dataset.replace('bffhq_', '')
        opts.percent = percent

        train_dataset = data.BFFHQ(root=opts.data_dir, split="train", percent=percent, transform=T_train)
        test_dataset = data.BFFHQ(root=opts.data_dir, split="test", percent=percent, transform=T_test)
        opts.n_classes = 2
    
    elif 'biased-mnist' in opts.dataset:
        rho = float(opts.dataset.replace('biased-mnist_', ''))
        opts.rho = rho

        train_dataset = data.BiasedMNIST(root=opts.data_dir, train=True, transform=T_train,
                                         download=True, data_label_correlation=rho)
        test_dataset = data.BiasedMNIST(root=opts.data_dir, train=False, transform=T_test,
                                        download=True, data_label_correlation=0.1)
        opts.n_classes = 10

    elif opts.dataset == 'imagenet100':
        train_dataset = data.ImageNet100(root=os.path.join(opts.data_dir, 'train'), transform=T_train)
        train_dataset = data.MapDataset(train_dataset, lambda x, y: (x, y, 0))
        print(len(train_dataset), 'training images')

        test_dataset = data.ImageNet100(root=os.path.join(opts.data_dir, 'val'), transform=T_test)
        test_dataset = data.MapDataset(test_dataset, lambda x, y: (x, y, 0))
        print(len(test_dataset), 'test images')
        opts.n_classes = 100
    
    elif opts.dataset == 'imagenet':
        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(opts.data_dir, 'train'), transform=T_train)
        train_dataset = data.MapDataset(train_dataset, lambda x, y: (x, y, 0))
        print(len(train_dataset), 'training images')

        test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(opts.data_dir, 'val'), transform=T_test)
        test_dataset = data.MapDataset(test_dataset, lambda x, y: (x, y, 0))
        print(len(test_dataset), 'test images')
        opts.n_classes = 1000
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True,
                                               num_workers=8, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, 
                                              persistent_workers=True)

    return train_loader, test_loader

def load_model(opts):
    if 'resnet' in opts.model:
        model = models.SupConResNet(opts.model, feat_dim=opts.feat_dim,
                                    num_classes=opts.n_classes,
                                    train_on_head=opts.train_on_head)
        
        if isinstance(model.encoder, torchvision.models.resnet.ResNet) and opts.dataset in ['cifar10', 'cifar100']:
            print("Adjusting first conv layer for cifar")
            model.encoder.conv1 = torch.nn.Conv2d(3, model.encoder.conv1.out_channels, 
                                                  kernel_size=3, stride=1, padding=1, bias=False)

    elif opts.model == 'simpleconvnet':
        model = models.SupConSimpleConvNet(feat_dim=opts.feat_dim,
                                           num_classes=opts.n_classes,
                                           train_on_head=opts.train_on_head)

    opts.feat_dim = model.feat_dim

    if opts.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)
    model = model.to(opts.device)

    if "infonce" in opts.method:
        if "strong" in opts.method:
            criterion = losses.EpsilonSupCon(temperature=opts.temp, form=opts.form, epsilon=opts.epsilon)
        else:
            criterion = losses.EpsilonSupInfoNCE(temperature=opts.temp, form=opts.form, epsilon=opts.epsilon)

    else:
        raise ValueError('Unsupported loss function', opts.method)

    criterion = criterion.to(opts.device)
    
    return model, criterion

def load_optimizer(model, criterion, opts):
    if torch.cuda.device_count() > 1:
        parameters = [{'params': model.module.encoder.parameters()},
                    {'params': model.module.head.parameters()}]
    else:
        parameters = [{'params': model.encoder.parameters()},
                    {'params': model.head.parameters()}]

    if opts.beta > 0:
        parameters = model.parameters()

    if "auto" in opts.method:
        parameters.append({'params': criterion.epsilon,
                           'lr': opts.lr_epsilon,
                           'weight_decay': 0})

    if opts.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=opts.lr, 
                                    momentum=opts.momentum,
                                    weight_decay=opts.weight_decay)
    elif opts.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=opts.lr, weight_decay=opts.weight_decay)

    else:
        optimizer = LARS(parameters, lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
        
    if opts.lr_decay == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.epochs, 
                                                               verbose=True)
    elif opts.lr_decay == 'step':
        milestones = [int(s) for s in opts.lr_decay_epochs.split(',')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, verbose=True)
    
    elif opts.lr_decay == 'none':
        scheduler = None

    optimizer_fc, scheduler_fc = None, None

    if opts.beta == 0:
        fc_params = model.fc.parameters() if torch.cuda.device_count() <= 1 else model.module.fc.parameters()
        if opts.mlp_optimizer == "sgd":
            optimizer_fc = torch.optim.SGD(fc_params, lr=opts.mlp_lr, momentum=0.9,
                                        weight_decay=0)
        elif opts.mlp_optimizer == "adam":
            optimizer_fc = torch.optim.Adam(fc_params, lr=opts.mlp_lr,
                                            weight_decay=0)
        
        if opts.mlp_lr_decay == 'cosine':
            scheduler_fc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fc, T_max=opts.epochs, 
                                                                    verbose=True)
        else:
            scheduler_fc = None

    print((optimizer, scheduler), (optimizer_fc, scheduler_fc))
    return (optimizer, scheduler), (optimizer_fc, scheduler_fc)

def train(train_loader, model, criterion, optimizers, opts, epoch, scaler=None):
    loss = AverageMeter()
    nce = AverageMeter()
    ce = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    optimizer, optimizer_fc = optimizers

    all_outputs, all_labels = [], []

    t1 = time.time()
    for idx, (images, labels, bias_labels) in enumerate(train_loader):
        data_time.update(time.time() - t1)

        images = torch.cat(images, dim=0)
        images, labels, bias_labels = images.to(opts.device), labels.to(opts.device), bias_labels.to(opts.device)
        bsz = labels.shape[0]
        
        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        with torch.set_grad_enabled(True):
            with torch.cuda.amp.autocast(scaler is not None):
                projected, feats, logits = model(images)
                
                projected = torch.split(projected, [bsz]*opts.n_views, dim=0)
                projected = torch.cat([f.unsqueeze(1) for f in projected], dim=1)

                feats = torch.split(feats, [bsz]*opts.n_views, dim=0)
                feats = torch.cat([f.unsqueeze(1) for f in feats], dim=1)

                logits = torch.split(logits, [bsz]*opts.n_views, dim=0)
                logits = torch.cat([f.unsqueeze(1) for f in logits], dim=1)

                running_nce = criterion(projected, feats[:, 0], logits, labels, bias_labels)
                running_ce = F.cross_entropy(logits[:, 0], labels)

                running_loss = running_nce
                if opts.beta > 0:
                    running_loss = running_nce + opts.beta*running_ce
          
        optimizer.zero_grad()

        if optimizer_fc is not None:
            optimizer_fc.zero_grad()

        if scaler is None:
            if optimizer_fc is not None:
                running_ce.backward(retain_graph=True) # Backward cross-entropy from last layer
                optimizer_fc.step()
                optimizer.zero_grad() # Stop-gradient on the encoder

            running_loss.backward() # Backward infonce loss on the encoder
            optimizer.step()
        else:
            if optimizer_fc is not None:
                scaler.scale(running_ce).backward(retain_graph=True)
                scaler.step(optimizer_fc)
                optimizer.zero_grad()

            scaler.scale(running_loss).backward()
            scaler.step(optimizer)
            
            scaler.update()
        
        loss.update(running_loss.item(), bsz)
        nce.update(running_nce.item(), bsz)
        ce.update(running_ce.item(), bsz)
        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(train_loader) - idx)

        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"NCE {nce.avg:.3f}\t"
                  f"CE {ce.avg:.3f}\t"
                  f"loss {loss.avg:.3f}\t")
        
        all_outputs.append(logits[:, 0].detach())
        all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    accuracy_train = accuracy(all_outputs, all_labels)[0]

    return loss.avg, accuracy_train, batch_time.avg, data_time.avg

def measure_similarity(feat, labels, bias_labels):
    bsz = feat.shape[0]

    if labels.shape[0] != bsz:
        raise ValueError('Num of labels does not match num of features')
    if bias_labels.shape[0] != bsz:
        raise ValueError('Num of bias_labels does not match num of features')

    similarity = torch.matmul(feat, feat.T)

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T)
    negative_mask = ~positive_mask

    bias_labels = bias_labels.view(-1, 1)
    aligned_mask = torch.eq(bias_labels, bias_labels.T)
    conflicting_mask = ~aligned_mask

    pos_aligned = positive_mask * aligned_mask
    aligned_sim = similarity * pos_aligned
    aligned_sim_mean = (aligned_sim.sum(dim=1) - 1) / (torch.count_nonzero(aligned_sim, dim=1)+1)

    pos_conflicting = positive_mask * conflicting_mask
    conflicting_sim = similarity * pos_conflicting
    conflicting_sim_mean = conflicting_sim.sum(dim=1) / (torch.count_nonzero(conflicting_sim, dim=1)+1)

    # print(conflicting_sim.sum(dim=1), torch.count_nonzero(conflicting_sim, dim=1))

    neg_aligned = negative_mask * aligned_mask
    negative_aligned_sim = similarity * neg_aligned
    negative_aligned_sim_mean = (negative_aligned_sim.sum(dim=1)) / (torch.count_nonzero(neg_aligned, dim=1)+1)
    
    neg_conflicting = negative_mask * conflicting_mask
    negative_conflicting_sim = similarity * neg_aligned
    negative_conflicting_sim_mean = (negative_conflicting_sim.sum(dim=1)) / (torch.count_nonzero(neg_conflicting, dim=1)+1)

    # print(f"pos-aligned: {pos_aligned.sum() - bsz}, pos-conflicting: {pos_conflicting.sum() - bsz}")
    return (aligned_sim[torch.nonzero(pos_aligned, as_tuple=True)], aligned_sim_mean.mean()), \
           (conflicting_sim[torch.nonzero(pos_conflicting, as_tuple=True)], conflicting_sim_mean.mean()), \
           (negative_aligned_sim[torch.nonzero(neg_aligned, as_tuple=True)], negative_aligned_sim_mean.mean()), \
           (negative_conflicting_sim[torch.nonzero(neg_conflicting, as_tuple=True)], negative_conflicting_sim_mean.mean())
           
def test(test_loader, model, criterion, opts):
    model.eval()

    loss = AverageMeter()
    all_outputs, all_labels = [], []

    aligned_sim, conflicting_sim, negative_aligned_sim, negative_conflicting_sim = [], [], [], []
    aligned_similarity = AverageMeter()
    conflicting_similarity = AverageMeter()
    negative_aligned_similarity = AverageMeter()
    negative_conflicting_similarity = AverageMeter()

    for images, labels, bias_labels in test_loader:
        images, labels = images.to(opts.device), labels.to(opts.device)
        bias_labels = bias_labels.to(opts.device)

        with torch.no_grad():
            projected, feats, logits = model(images)
            running_loss = criterion(projected[:, None], feats, logits, labels, bias_labels)
            (al_sim, al_sim_mean), (con_sim, con_sim_mean), \
            (neg_al_sim, neg_al_sim_mean), (neg_confl_sim, neg_confl_sim_mean) \
                = measure_similarity(projected, labels, bias_labels)
        
        loss.update(running_loss.item(), images.shape[0])
        
        all_outputs.append(logits.detach())
        all_labels.append(labels)

        if not torch.isinf(al_sim_mean):
            aligned_similarity.update(al_sim_mean.item(), images.shape[0])
            aligned_sim.append(al_sim.view(-1))

        if not torch.isinf(con_sim_mean):
            conflicting_similarity.update(con_sim_mean.item(), images.shape[0])
            conflicting_sim.append(con_sim.view(-1))
        
        if not torch.isinf(neg_al_sim_mean):
            negative_aligned_similarity.update(neg_al_sim_mean.item(), images.shape[0])
            negative_aligned_sim.append(neg_al_sim.view(-1))

        if not torch.isinf(neg_confl_sim_mean):
            negative_conflicting_similarity.update(neg_confl_sim_mean.item(), images.shape[0])
            negative_conflicting_sim.append(neg_confl_sim.view(-1))

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    aligned_sim = torch.cat(aligned_sim)
    conflicting_sim = torch.cat(conflicting_sim)
    negative_aligned_sim = torch.cat(negative_aligned_sim)
    negative_conflicting_sim = torch.cat(negative_conflicting_sim)

    accuracy_test = accuracy(all_outputs, all_labels)[0]
    print(f'FC test accuracy: {accuracy_test:.2f}')

    return loss.avg, accuracy_test, (aligned_sim, aligned_similarity.avg), \
           (conflicting_sim, conflicting_similarity.avg), \
           (negative_aligned_sim, negative_aligned_similarity.avg), \
           (negative_conflicting_sim, negative_conflicting_similarity.avg)
        
def test_mlp(train_loader, test_loader, model, opts):
    model.eval()
    print("training linear classifier from scratch..")
    train_features, train_labels = [], []
    for images, labels, bias_labels in train_loader:
        images, labels = images.to(opts.device), labels.to(opts.device)

        with torch.no_grad():
            projected, features, _ = model(images)
        
        if opts.train_on_head:
            train_features.append(projected)
        else:
            train_features.append(features)

        train_labels.append(labels)
    
    train_features = torch.cat(train_features)
    train_labels = torch.cat(train_labels)
    print('training features:', train_features.shape, train_labels.shape)

    clf = models.MLPClassifier(hidden_layer_sizes=(), 
                               learning_rate_init=0.001,
                               max_iter=500,
                               batch_size=2048,
                               solver='adam')
    clf.fit(train_features, train_labels)
   
    accuracy_train = clf.score(train_features, train_labels)*100
    print(f'train accuracy: {accuracy_train:.2f}')

    test_features, test_labels = [], []
    for images, labels, _ in test_loader:
        images, labels = images.to(opts.device), labels.to(opts.device)

        with torch.no_grad():
            projected, features, _ = model(images)
        
        if opts.train_on_head:
            test_features.append(projected)
        else:
            test_features.append(features)

        test_labels.append(labels)

    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels)
    print('test features:', test_features.shape, test_labels.shape)

    accuracy_test = clf.score(test_features, test_labels)*100
    return accuracy_train, accuracy_test

if __name__ == '__main__':
    opts = parse_arguments()
    set_seed(opts.trial)

    train_loader, test_loader = load_data(opts)
    model, infonce = load_model(opts)
    (optimizer, scheduler), (optimizer_fc, scheduler_fc) = load_optimizer(model, infonce, opts)
    
    if opts.batch_size > 256:
        opts.warm = True
    
    if opts.warm:
        opts.warm_epochs = 10
        opts.warmup_from = 0.01
        opts.model = f"{opts.model}_warm"
        
        if opts.lr_decay == 'cosine':
            eta_min = opts.lr * (0.1 ** 3)
            opts.warmup_to = eta_min + (opts.lr - eta_min) * (1 + math.cos(math.pi * opts.warm_epochs / opts.epochs)) / 2
        else:
            opts.warmup_to = opts.lr

    ensure_dir(opts.log_dir)
    method = opts.method
    if opts.selfsup:
        method = f"{method}_self"
    if opts.augplus:
        method = f"{method}_aug+"

    run_name = (f"{method}_{opts.form}_{opts.dataset}_{opts.model}_"
                f"{opts.optimizer}_bsz{opts.batch_size}_"
                f"lr{opts.lr}_{opts.lr_decay}_t{opts.temp}_eps{opts.epsilon}_"
                f"lr-eps{opts.lr_epsilon}_feat{opts.feat_dim}_"
                f"{'identity_' if opts.train_on_head else 'head_'}"
                f"alpha{opts.alpha}_beta{opts.beta}_lambda{opts.lambd}_kld{opts.kld}_{opts.dist}_"
                f"mlp_lr{opts.mlp_lr}_mlp_optimizer_{opts.mlp_optimizer}_"
                f"trial{opts.trial}")
    tb_dir = os.path.join(opts.log_dir, run_name)
    opts.model_class = model.__class__.__name__
    opts.criterion = infonce
    opts.optimizer_class = optimizer.__class__.__name__
    opts.scheduler = scheduler.__class__.__name__ if scheduler is not None else None

    wandb.init(project="contrastive-learning-debiasing", config=opts, name=run_name, sync_tensorboard=True)
    print('Config:', opts)
    print('Model:', model)
    print('Criterion:', infonce)
    print('Optimizer:', optimizer)
    print('Scheduler:', scheduler)

    writer = torch.utils.tensorboard.writer.SummaryWriter(tb_dir)
    
    def target_loss(projected, labels):
        if opts.selfsup:
            return opts.alpha*infonce(projected)
        return opts.alpha*infonce(projected, labels)
    criterion = lambda projected, feats, logits, labels, bias_labels: target_loss(projected, labels)
            
    if opts.lambd != 0:
        print("Applying regularization")
        
        def infonce_fairkl(projected, feats, logits, labels, bias_labels):
            feats = F.normalize(feats)
            return opts.alpha * target_loss(projected, labels) + \
                   opts.lambd * losses.fairkl(feats, labels, bias_labels, 1.0, kld=opts.kld)
        
        criterion = infonce_fairkl

    scaler = torch.cuda.amp.GradScaler() if opts.amp else None
    if opts.amp:
        print("Using AMP")
    
    start_time = time.time()
    best_acc = 0.
    for epoch in range(1, opts.epochs + 1):
        t1 = time.time()
        loss_train, accuracy_train, batch_time, data_time = train(train_loader, model, criterion, (optimizer, optimizer_fc), opts, epoch, scaler)
        t2 = time.time()

        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("train/loss", loss_train, epoch)
        writer.add_scalar("train/acc@1", accuracy_train, epoch)
        if "auto" in opts.method:
            writer.add_scalar("train/epsilon", infonce.epsilon, epoch)

        writer.add_scalar("BT", batch_time, epoch)
        writer.add_scalar("DT", data_time, epoch)
        print(f"epoch {epoch}, total time {t2-start_time:.2f}, epoch time {t2-t1:.3f} "
              f"acc {accuracy_train:.2f} loss {loss_train:.4f}")
        
        if scheduler is not None:
            scheduler.step()

        if (epoch % opts.test_freq == 0) or epoch == 1 or epoch == opts.epochs:
            loss_test, accuracy_test, aligned_sim, conflicting_sim, \
            negative_aligned_sim, negative_conflicting_sim \
                 = test(test_loader, model, criterion, opts)
            writer.add_scalar("test/loss", loss_test, epoch)
            writer.add_scalar("test/acc@1", accuracy_test, epoch)
            print(f"test accuracy {accuracy_test:.2f}")

            print(f"""pos-aligned sim {aligned_sim[1]:.4f}, pos-conflict sim {conflicting_sim[1]:.4f}, """
                  f"""neg-aligned sim {negative_aligned_sim[1]:.4f} neg-conflict sim {negative_conflicting_sim[1]:.4f}""")
            writer.add_scalar('test/aligned_sim_mean', aligned_sim[1], epoch)
            writer.add_scalar('test/conflicting_sim_mean', conflicting_sim[1], epoch)
            writer.add_scalar('test/negative_aligned_sim_mean', negative_aligned_sim[1])
            writer.add_scalar('test/negative_conflicting_sim_mean', negative_conflicting_sim[1])

            try:
                writer.add_histogram('test/aligned_sim', aligned_sim[0], epoch, bins=256, max_bins=512)
                writer.add_histogram('test/conflicting_sim', conflicting_sim[0], epoch, bins=256, max_bins=512)
                writer.add_histogram('test/negative_aligned_sim', negative_aligned_sim[0], epoch, bins=256, max_bins=512)
                writer.add_histogram('test/negative_conflicting_sim', negative_conflicting_sim[0], epoch, bins=256, max_bins=512)
            except:
                pass

            if accuracy_test > best_acc:
                best_acc = accuracy_test
    
        writer.add_scalar("best_acc@1", best_acc, epoch)
    
    print(f"best accuracy: {best_acc:.2f}")