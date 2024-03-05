import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel

class Electra(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, maps=None) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        # self.article_class_num = len(maps["article2idx"])
        self.hid_dim = hid_dim

        self.electra = AutoModel.from_pretrained("electra-small") 
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # self.fc_article = nn.Linear(self.hid_dim, self.article_class_num)
        self.fc_charge = nn.Linear(self.hid_dim, self.charge_class_num)
        # self.fc_judge = nn.Linear(self.hid_dim, 1)

        self.dropout = nn.Dropout(0.4)

    def forward(self, data):
        text = data["justice"]["input_ids"].cuda()
        x = self.electra(text)
        out = x.last_hidden_state[:,0] # [256]

        out_charge = self.fc_charge(out)
        # out_article = self.fc_article(out)
        # out_judge = self.fc_judge(out)
        return {
            # "article": out_article,
            "charge": out_charge,
            # "judge": out_judge
        }
