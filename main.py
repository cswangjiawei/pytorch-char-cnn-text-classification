import argparse
from util import load_datasets
import time
from model import CharCNN
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from train import train_model, evaluate
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import random
import numpy as np


seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Character-level Convolutional Networks for Text Classification')
    parser.add_argument('--train_data_path', default='data/ag_news_csv/train.csv')
    parser.add_argument('--test_data_path', default='data/ag_news_csv/test.csv')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--savedir', default='save_model')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam'], default='SGD')
    parser.add_argument('--epochs', default=20)
    parser.add_argument('--patience', default=10)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--length', default=1014, help='the length of input feature')
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--lower', default=True)

    args = parser.parse_args()
    dataloader = load_datasets(args)
    best_error = 1000
    early_stop = 0

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    model_name = args.savedir + '/' + 'best.pt'
    train_begin = time.time()
    print('train begin', '-'*50)
    print()
    print()
    model = CharCNN(70, args.dropout)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    else:
        print('optimizer is bad')
        optimizer = None
        exit(0)

    writer = SummaryWriter('log')

    for epoch in range(args.epochs):
        epoch_begin = time.time()
        print('train {}/{} epoch'.format(epoch+1, args.epochs))
        train_loss = train_model(dataloader['train_dataloader'], model, criterion, optimizer)
        print('train_loss:', train_loss)
        writer.add_scalar('loss', train_loss, epoch)

        test_error = evaluate(dataloader['dev_dataloader'], model)
        print('testing error:', test_error)
        writer.add_scalar('test_error', test_error, epoch)

        if args.optimizer == 'SGD':
            scheduler.step()

        if test_error < best_error:
            early_stop = 0
            best_error = test_error
            torch.save(model.state_dict(), model_name)

        else:
            early_stop += 1

        epoch_end = time.time()
        cost_time = epoch_end - epoch_begin
        print('train {}th epoch cost {}m {}s'.format(epoch + 1, int(cost_time/60), int(cost_time % 60)))
        print()

        if early_stop >= args.patience:
            exit(0)

    train_end = time.time()
    train_cost = train_end - train_begin
    hour = int(train_cost / 3600)
    min = int((train_cost % 3600) / 60)
    second = int(train_cost % 3600 % 60)
    print()
    print()
    print('train end', '-'*50)
    print('train total cost {}h {}m {}s'.format(hour, min, second))
