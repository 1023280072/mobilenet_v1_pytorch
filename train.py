import argparse
import os
import json
import torch
import shutil
from torch import optim
from torch import nn
from tqdm import tqdm
from dataset import DogvsCatDataset
from model import MobileNet
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--work_dir', type=str, default='work_dir')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--optim', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.work_dir):
        shutil.rmtree(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    tensorboard_log_dir = os.path.join(args.work_dir, 'runs')
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    log_json = os.path.join(args.work_dir, 'log.json')
    log_json_f = open(log_json, 'a')
    checkpoints_dir = os.path.join(args.work_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    train_dataset = DogvsCatDataset('dogvscat/train')
    val_dataset = DogvsCatDataset('dogvscat/val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    net = MobileNet()
    loss_func = nn.BCEWithLogitsLoss()
    if args.cuda:
        net = net.cuda()
        loss_func = loss_func.cuda()

    if args.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    for epoch in range(1, args.max_epoch + 1):
        adjust_optim(args.optim, optimizer, epoch)
        train_one_epoch(args, epoch, net, loss_func, train_dataloader, optimizer, writer, log_json_f)
        test(args, epoch, net, loss_func, val_dataloader, writer, log_json_f, 'val')
        torch.save(net, os.path.join(checkpoints_dir, 'epoch_' + str(epoch) + '.pth'))

    log_json_f.close()
    writer.close()

def adjust_optim(optim_type, optimizer, epoch):
    if optim_type == 'sgd':
        if epoch < 150: lr = 0.1
        elif epoch == 150: lr = 0.01
        elif epoch == 225: lr = 0.001
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train_one_epoch(args, epoch, net, loss_func, train_dataloader, optimizer, writer, f):
    print('----------Training----------')
    net.train()
    num_processed = 0
    num_train = len(train_dataloader.dataset)
    for batch_idx, (img, label) in enumerate(train_dataloader):
        if args.cuda:
            img, label = img.cuda(), label.cuda()
        img, label = Variable(img), Variable(label).reshape(-1, 1).type(torch.float)
        optimizer.zero_grad()
        pred = net(img)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
        num_processed += len(img)
        train_iter = batch_idx + (epoch - 1) * len(train_dataloader)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, num_processed, num_train, 100. * num_processed / num_train, loss.data))
            tem_dict = {'mode': 'train', 'iter': train_iter, 'loss': float(loss.data)}
            json.dump(tem_dict, f)
            f.write('\n')
            writer.add_scalar('train_loss', float(loss.data), train_iter)

def test(args, epoch, net, loss_func, dataloader, writer, f, name):
    print('----------Testing----------')
    net.eval()
    loss = 0.
    acc = 0.
    err = 0.
    for img, label in tqdm(dataloader):
        with torch.no_grad():
            if args.cuda:
                img, label = img.cuda(), label.cuda()
            img, label = Variable(img), Variable(label).reshape(-1, 1).type(torch.float)
            pred = net(img)
            loss += loss_func(pred, label)
            pred = (pred > 0).type(torch.float)
            err += pred.ne(label.data).cpu().sum()
    loss, err = float(loss), float(err)
    loss /= len(dataloader)
    err /= len(dataloader.dataset)
    acc = 1. - err
    print('Test ' + name + ' dataset: Average loss: {:.6f}, Accuracy: {:.4f}'.format(loss, acc))
    tem_dict = {'mode': 'test', 'name': name, 'epoch': epoch, 'loss': loss, 'acc': acc}
    json.dump(tem_dict, f)
    f.write('\n')
    writer.add_scalar(name + '_dataset_loss', loss, epoch)
    writer.add_scalar(name + '_dataset_accuracy', acc, epoch)



if __name__ == '__main__':
    main()


