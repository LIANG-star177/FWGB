import pickle
from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
import random
import os
import jieba
import re
from tqdm import tqdm
from transformers import BertTokenizer
import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self,vocab_size=5000,emb_dim=128,hid_dim=128,maps=None) -> None:
        super(TextRNN,self).__init__()
        self.vocab_size=vocab_size
        self.emb_dim=emb_dim
        self.hid_dim=hid_dim
        self.charge_class_num = len(maps["charge2idx"])

        self.embedding=nn.Embedding(vocab_size,emb_dim)
        self.lstm=nn.LSTM(emb_dim,hid_dim,bidirectional=True,batch_first=True,dropout=0.5)
        self.w=nn.Parameter(torch.zeros(hid_dim*2))
        self.fc_input_dim=hid_dim*2
        self.fc1=nn.Linear(self.fc_input_dim,self.hid_dim)
        self.fc2=nn.Linear(self.hid_dim,self.charge_class_num)
        self.dropout=nn.Dropout(0.4)

    def forward(self,data):
        text=data["justice"]["input_ids"].cuda()
        x=self.embedding(text)
        hiddens,_=self.lstm(x) #[64,32,256]batch_size,seq_len,hidden
        out=hiddens[:,-1,:]#取最后一个hidden
        out=self.fc1(out)
        out=nn.ReLU()(out)
        out=self.fc2(out)
        return {"charge":out}

# net=TextRNN(5000,128,128,10)
# print(net)