from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
from numpy import *
from dataset import train_dataset
from SegNet import Segnet

parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', type=int, default=1)
parse.add_argument('--epochs', type=int, default=10)
parse.add_argument('--cuda', type=bool, default=False)
parse.add_argument('--num_GPU', type=int, default=1)
parse.add_argument('--num_workers', type=int, default=2)
parse.add_argument('--gpu_device', type=str, default='cuda:0')
parse.add_argument('--input_num', type=int, default=3)
parse.add_argument('--label_num', type=int, default=12)
parse.add_argument('--model_save_path', type=str, default='model')
args = parse.parse_args()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


args.manual_seed = random.randint(1, 10000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)

train_datatset = train_dataset('dataset/CamVid')
train_loader = torch.utils.data.DataLoader(dataset=train_datatset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers)

device = torch.device(args.gpu_device if args.cuda else "cpu")
net = Segnet(args.input_num, args.label_num)
net.apply(weights_init)
if args.num_GPU > 1:
    net = nn.DataParallel(net)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

if __name__ == '__main__':
    net.train()
    for epoch in range(1, args.epochs):
        i = 0
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = net(img)
            loss = criterion(output, torch.squeeze(label, 1).long())
            loss.backward()
            optimizer.step()
            i += 1
            print('[%d][%d] Loss: %.4f' % (epoch, i, loss.item()))

    torch.save(net.state_dict(), args.model_save_path + 'segnet.pth')
