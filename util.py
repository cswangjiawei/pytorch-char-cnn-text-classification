from torch.utils.data import Dataset
import csv
import torch
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, data_path, lower, length):
        super(MyDataset, self).__init__()
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
        self.data_path = data_path
        self.lower = lower
        self.length =length
        self.data, self.label = self.load_data()

    def __getitem__(self, item):
        sent_tensor = self.get_data_represent(item)
        label_tensor = torch.tensor(self.label[item])
        return {'sent': sent_tensor, 'label': label_tensor}

    def __len__(self):
        return len(self.label)

    def get_data_represent(self, item):
        sent = self.data[item]
        sent_tensor = torch.zeros(self.length).long()
        for i, char in enumerate(sent):
            if i == self.length:
                break
            alphabet_index = self.alphabet.find(char)
            if alphabet_index != -1:
                sent_tensor[i] = alphabet_index

        return sent_tensor

    def load_data(self):
        data = []
        label = []
        with open(self.data_path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in csv_reader:
                text = ' '.join(row[1:])
                if self.lower:
                    text = text.lower()
                data.append(text)
                label.append(int(row[0]))
        return data, label


def load_datasets(args):
    train_dataset = MyDataset(args.train_data_path, args.lower, args.length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    test_dataset = MyDataset(args.test_data_path, args.lower, args.length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    return {'train_dataloader': train_dataloader, 'dev_dataloader':  test_dataloader}