import torch.nn as nn
import torch


class LSTM_attn(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=256, hid_dim=256, maps=None) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        # self.article_class_num = len(maps["article2idx"])

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, bidirectional=True,
                            batch_first=True, dropout=0.5)

        self.w = nn.Parameter(torch.zeros(hid_dim * 2))
        self.fc_input_dim = hid_dim*2
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.fc_input_dim, self.hid_dim)
        # self.fc_article = nn.Linear(self.hid_dim, self.article_class_num)
        self.fc_charge = nn.Linear(self.hid_dim, self.charge_class_num)
        # self.fc_judge = nn.Linear(self.hid_dim, 1)

    def forward(self, data):
        text = data["justice"]["input_ids"].cuda()
        # text = data["rationale"].cuda()
        x = self.embedding(text)
        hidden, _ = self.lstm(x)  # [64, 512, 256]

        mat = nn.Tanh()(hidden)  # [64, 512, 256]
        alpha = nn.Softmax(dim=1)(torch.matmul(mat, self.w)
                                  ).unsqueeze(-1)  # [64, 512, 1]
        out = hidden * alpha  # [64, 512, 256]
        out = torch.sum(out, dim=1)  # [64, 256]
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out_charge = self.fc_charge(out)
        # out_article = self.fc_article(out)
        # out_judge = self.fc_judge(out)
        return {
            # "article": out_article,
            "charge": out_charge
            # "judge": out_judge
        },alpha
