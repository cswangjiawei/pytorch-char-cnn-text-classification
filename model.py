import torch.nn as nn
import torch


def Conv1d(in_channel, out_channel, kernel_size):
    m = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
    nn.init.normal_(m.weight, mean=0, std=0.05)
    return nn.utils.weight_norm(m)


class CharCNN(nn.Module):
    def __init__(self, vocab_size, dropout):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, vocab_size, padding_idx=0)
        self.embedding.weight.data[1:].copy_(self.random_embedding(vocab_size))
        self.embedding.weight.requires_grad = False
        self.feature_dim = vocab_size

        self.cnn_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()

        self.conv1 = nn.Sequential(
            Conv1d(self.feature_dim, 256, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.cnn_list.append(self.conv1)

        self.conv2 = nn.Sequential(
            Conv1d(256, 256, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.cnn_list.append(self.conv2)

        self.conv3 = nn.Sequential(
            Conv1d(256, 256, kernel_size=3),
            nn.ReLU()
        )
        self.cnn_list.append(self.conv3)
        self.cnn_list.append(self.conv3)
        self.cnn_list.append(self.conv3)

        self.conv6 = nn.Sequential(
            Conv1d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.cnn_list.append(self.conv6)

        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc_list.append(self.fc1)

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc_list.append(self.fc2)

        self.fc3 = nn.Linear(1024, 4)
        self.fc_list.append(self.fc3)

    @staticmethod
    def random_embedding(vocab_size):
        embedding = torch.eye(vocab_size)
        return embedding

    def forward(self, word_input):
        word_represent = self.embedding(word_input)
        cnn_feature = word_represent.transpose(1, 2)
        for conv in self.cnn_list:
            cnn_feature = conv(cnn_feature)

        fc_feature = cnn_feature.view(cnn_feature.size(0), -1)

        for fc in self.fc_list:
            fc_feature = fc(fc_feature)

        return fc_feature