# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DPCNN(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=128, hid_dim=128, maps=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        # self.article_class_num = len(maps["article2idx"])
        self.num_filters = emb_dim
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.conv_region = nn.Conv2d(
            1, self.num_filters, (3, self.hid_dim), stride=1)
        self.conv = nn.Conv2d(
            self.num_filters, self.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        # self.fc_article = nn.Linear(self.num_filters, self.article_class_num)
        self.fc_charge = nn.Linear(self.num_filters, self.charge_class_num)
        # self.fc_judge = nn.Linear(self.num_filters, 1)

    def forward(self, data):
        x = data["justice"]["input_ids"].cuda()
        x = self.embedding(x)
        # x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x_charge = self.fc_charge(x)
        # x_article = self.fc_article(x)
        # x_judge = self.fc_judge(x)
        return {
            # "article": x_article,
            "charge": x_charge,
            # "judge": x_judge
        }

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
