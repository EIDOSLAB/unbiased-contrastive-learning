import os
from sklearn import feature_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import argparse
import data
import models
import losses
import time
import wandb
import torch.utils.tensorboard

from torchvision import transforms
from torchvision import datasets
from util import AverageMeter, accuracy, ensure_dir, set_seed
from main_infonce import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Contrastive debiasing",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
    parser.add_argument('--log_dir', type=str, help='tensorboard log dir', default='logs')

    parser.add_argument('--data_dir', type=str, help='path of data dir', required=True)
    parser.add_argument('--dataset', type=str, help='dataset (format name_attr e.g. biased-mnist_0.999)', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)

    parser.add_argument('--epochs', type=int, help='number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step', 'none'], default='cosine')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="100,150")
    parser.add_argument('--optimizer', type=str, help="optimizer (adam or sgd)", choices=["adam", "sgd"], default="sgd")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-4)
    parser.add_argument('--alpha', type=float, help='CE weight', default=1.)
    parser.add_argument('--lambd', type=float, help='lagrangian weight for debiasing', default=0.)
    parser.add_argument('--kld', type=float, help='weight of std term', default=1.)
    parser.add_argument('--temp', type=float, help='sim temperature (for lagrange)', default=1.0)

    parser.add_argument('--test_freq', type=int, help='test frequency', default=50)
    parser.add_argument('--model', type=str, help='model architecture')
    parser.add_argument('--amp', action='store_true', help='use amp')

    return parser.parse_args()

def load_model(opts):
    normalize = opts.lambd != 0

    if 'resnet' in opts.model:
        model = models.CEResNet(opts.model, num_classes=opts.n_classes,
                                normalize=normalize)
        if normalize:
            model.encoder.layer4[1].relu = nn.Tanh()
    
    elif opts.model == 'simpleconvnet':
        model = models.CESimpleConvNet(num_classes=opts.n_classes,
                                       normalize=normalize)
        if normalize:
            model.encoder.relu = nn.Tanh()
    
    model = model.to(opts.device)
    criterion = F.cross_entropy
    
    return model, criterion

def load_optimizer(model, criterion, opts):
    parameters = [{'params': model.parameters()}]

    if opts.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=opts.lr, 
                                    momentum=opts.momentum,
                                    weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.Adam(parameters, lr=opts.lr, weight_decay=opts.weight_decay)

    if opts.lr_decay == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.epochs, 
                                                               verbose=True)
    elif opts.lr_decay == 'step':
        milestones = [int(s) for s in opts.lr_decay_epochs.split(',')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, verbose=True)
    
    elif opts.lr_decay == 'none':
        scheduler = None
    
    return optimizer, scheduler

def train(model, train_loader, criterion, optimizer, opts, epoch, scaler=None):
    loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    all_outputs, all_labels = [], []

    t1 = time.time()
    for idx, (images, labels, bias_labels) in enumerate(train_loader):
        data_time.update(time.time() - t1)

        images, labels, bias_labels = images.to(opts.device), labels.to(opts.device), bias_labels.to(opts.device)
        bsz = images.shape[0]

        with torch.set_grad_enabled(True):
            with torch.cuda.amp.autocast(scaler is not None):
                logits, feats = model(images)
                running_loss = criterion(logits, labels, feats, bias_labels)
        
        optimizer.zero_grad()
        if scaler is None:
            running_loss.backward()
            optimizer.step()
        else:
            scaler.scale(running_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss.update(running_loss.item(), bsz)
        all_outputs.append(logits.detach())
        all_labels.append(labels)

        batch_time.update(time.time() - t1)
        t1 = time.time()

        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f} ({batch_time.val:.3f})\t"
                  f"DT {data_time.avg:.3f} ({data_time.val:.3f})\t"
                  f"loss {loss.avg:.3f} ({loss.val:.3f})")
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    top1 = accuracy(all_outputs, all_labels, topk=(1,))[0]

    return loss.avg, top1, batch_time.avg, data_time.avg

def test(model, test_loader, criterion, opts):
    loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    all_outputs, all_labels = [], []

    t1 = time.time()
    for idx, (images, labels, bias_labels) in enumerate(test_loader):
        data_time.update(time.time() - t1)

        images, labels, bias_labels = images.to(opts.device), labels.to(opts.device), bias_labels.to(opts.device)
        bsz = images.shape[0]

        with torch.no_grad():
            logits, feats = model(images)
            running_loss = criterion(logits, labels, feats, bias_labels)
        
        loss.update(running_loss.item(), bsz)
        all_outputs.append(logits.detach())
        all_labels.append(labels)

        batch_time.update(time.time() - t1)
        t1 = time.time()

        if (idx + 1) % opts.print_freq == 0:
            print(f"Test: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f} ({batch_time.val:.3f})\t"
                  f"DT {data_time.avg:.3f} ({data_time.val:.3f})\t"
                  f"loss {loss.avg:.3f} ({loss.val:.3f})")
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    top1 = accuracy(all_outputs, all_labels, topk=(1,))[0]

    return loss.avg, top1, batch_time.avg, data_time.avg
        

if __name__ == '__main__':
    opts = parse_arguments()
    set_seed(opts.trial)

    train_loader, test_loader = load_data(opts)
    model, cross_entropy = load_model(opts)
    optimizer, scheduler = load_optimizer(model, cross_entropy, opts)

    ensure_dir(opts.log_dir)
    run_name = (f"CE_{opts.dataset}_{opts.model}_"
                f"bsz{opts.batch_size}_lr{opts.lr}_"
                f"alpha_{opts.alpha}_lambda{opts.lambd}_kld{opts.kld}_"
                f"trial{opts.trial}")
    tb_dir = os.path.join(opts.log_dir, run_name)
    opts.model_class = model.__class__.__name__
    opts.criterion = cross_entropy
    opts.optimizer_class = optimizer.__class__.__name__
    opts.scheduler = scheduler.__class__.__name__ if scheduler is not None else None
    wandb.init(project="contrastive-learning-debiasing", config=opts, name=run_name, sync_tensorboard=True)
    
    print('Model:', model.__class__.__name__)
    print('Criterion:', cross_entropy)
    print('Optimizer:', optimizer)
    print('Scheduler:', scheduler)
    
    writer = torch.utils.tensorboard.writer.SummaryWriter(tb_dir)

    criterion = lambda logits, labels, feats, bias_labels: cross_entropy(logits, labels)
    if opts.lambd != 0:
        print("Applying lagrangian")
        def regularized_ce(logits, labels, feats, bias_labels):
            feats = F.normalize(feats)
            return opts.alpha * cross_entropy(logits, labels) + \
                   opts.lambd * losses.fairkl(feats, labels, 
                                              bias_labels, 1.0,
                                              kld=opts.kld)
        criterion = regularized_ce
    
    scaler = torch.cuda.amp.GradScaler() if opts.amp else None
    start_time = time.time()
    best_acc = 0.

    for epoch in range(1, opts.epochs + 1):
        t1 = time.time()
        train_loss, top1, batch_time, data_time = train(model, train_loader, criterion, 
                                                              optimizer, opts, epoch, scaler=scaler)
        t2 = time.time()

        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc@1", top1, epoch)
        #writer.add_scalar("train/acc@5", top5, epoch)

        writer.add_scalar("BT", batch_time, epoch)
        writer.add_scalar("DT", data_time, epoch)
        print(f"epoch {epoch}, total time {t2-start_time:.2f}, epoch time {t2-t1:.3f}, loss {train_loss:.4f} "
              f"acc@1: {top1:.2f}")

        if scheduler is not None:
            scheduler.step()

        test_loss, top1, batch_time, data_time = test(model, test_loader, criterion, opts)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/acc@1", top1, epoch)
        # writer.add_scalar("test/acc@5", top5, epoch)
        print(f"test acc@1: {top1:.2f}") #  acc@5: {top5:.2f}")

        if top1 > best_acc:
            best_acc = top1
    
        writer.add_scalar("best_acc@1", best_acc, epoch)
    print(f"best accuracy: {best_acc:.2f}")