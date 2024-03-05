import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support


def log_distance_accuracy_function(outputs, label):
    # 128：batch size
    # 450应该是最大刑期30年，将outputs限幅到0~30年
    return float(torch.sum(torch.log(torch.abs(torch.clamp(outputs, 0, 450) - torch.clamp(label, 0, 450)) + 1)))


def log_square_loss(outputs, labels):
    return torch.mean((torch.log(torch.clamp(outputs, 0, 450) + 1) - torch.log(torch.clamp(labels, 0, 450) + 1)) ** 2)


class LJPPredictor(nn.Module):
    def __init__(self, hidden_dim, maps):
        super(LJPPredictor, self).__init__()

        self.hidden_size = hidden_dim

        charge_class_num = len(maps["charge2idx"])
        # article_class_num = len(maps["article2idx"])

        self.charge_fc = nn.Linear(self.hidden_size, charge_class_num)
        # self.article_fc = nn.Linear(self.hidden_size, article_class_num)
        # self.judge_fc = nn.Linear(self.hidden_size, 1)

    def forward(self, h):
        charge = self.charge_fc(h)
        # article = self.article_fc(h)
        # judge = self.judge_fc(h)
        # return {"charge": charge, "article": article, "judge": judge}
        return {"charge": charge}


class CNNEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(CNNEncoder, self).__init__()

        self.emb_dim = emb_dim
        self.output_dim = self.emb_dim // 4

        self.min_gram = 2
        self.max_gram = 5
        self.convs = []
        for a in range(self.min_gram, self.max_gram + 1):
            self.convs.append(nn.Conv2d(1, self.output_dim, (a, self.emb_dim)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = self.emb_dim
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]

        # [128, 1, 4096, 768] [batch, 1, seq len, hid dim]
        x = x.view(batch_size, 1, -1, self.emb_dim)

        conv_out = []
        gram = self.min_gram
        for conv in self.convs:
            y = self.relu(conv(x))
            y = torch.max(y, dim=2)[0].view(batch_size, -1)

            conv_out.append(y)
            gram += 1
        # len(conv_out): 4, kernel_size:2~5
        conv_out = torch.cat(conv_out, dim=1)

        return conv_out  # [batch, dim]


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMDecoder, self).__init__()
        self.feature_len = hidden_dim

        features = self.feature_len
        self.hidden_dim = features

        # self.task_name = ["article", "charge", "judge"]
        self.task_name = ["charge"]

        self.midfc = []
        for x in self.task_name:
            self.midfc.append(nn.Linear(features, features))

        self.cell_list = [None]
        for x in self.task_name:
            self.cell_list.append(nn.LSTMCell(
                self.feature_len, self.feature_len))

        self.hidden_state_fc_list = []
        for a in range(0, len(self.task_name) + 1):
            arr = []
            for b in range(0, len(self.task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []

        for a in range(0, len(self.task_name) + 1):
            arr = []
            for b in range(0, len(self.task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)

        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)

    def init_hidden(self, bs):
        self.hidden_list = []
        for a in range(0, len(self.task_name) + 1):
            self.hidden_list.append((torch.autograd.Variable(torch.zeros(bs, self.hidden_dim).cuda()),
                                     torch.autograd.Variable(torch.zeros(bs, self.hidden_dim).cuda())))

    def forward(self, x):
        fc_input = x  # x: [128, 768]
        outputs = {}
        batch_size = x.size()[0]
        self.init_hidden(batch_size)

        first = []
        for a in range(0, len(self.task_name) + 1):
            first.append(True)
        for a in range(1, len(self.task_name) + 1):
            h, c = self.cell_list[a](fc_input, self.hidden_list[a])
            for b in range(1, len(self.task_name) + 1):
                hp, cp = self.hidden_list[b]
                if first[b]:
                    first[b] = False
                    hp, cp = h, c
                else:
                    hp = hp + self.hidden_state_fc_list[a][b](h)
                    cp = cp + self.cell_state_fc_list[a][b](c)
                self.hidden_list[b] = (hp, cp)
            outputs[self.task_name[a - 1]
                    ] = self.midfc[a - 1](h).view(batch_size, -1)

        return outputs


class TopJudge(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, maps):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = CNNEncoder(emb_dim)
        self.decoder = LSTMDecoder(hid_dim)

        self.fc = LJPPredictor(hid_dim, maps)
        self.dropout = nn.Dropout(0.4)

        self.criterion = {
            # "article": nn.CrossEntropyLoss(),
            "charge": nn.CrossEntropyLoss(),
            # "judge": log_square_loss
        }
        self.accuracy_function = {
            # "article": f1_score,
            "charge": f1_score,
            # "judge": log_distance_accuracy_function,
        }

    def forward(self, data):
        x = data["justice"]["input_ids"]  # [128, 512] 512: max seq len
        x = self.embedding(x)  # [batch, seqlen, dim]
        hidden = self.encoder(x)  # [128, 128]
        hidden = self.dropout(hidden)  # dropout: 0.4
        out = self.decoder(hidden)
        for name in out:
            out[name] = self.fc(self.dropout(out[name]))[name]

        return out
