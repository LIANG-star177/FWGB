#!/usr/bin/env python
# coding: utf-8
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fmg
from collections import defaultdict
from gensim.models import Word2Vec

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


def keywords_id(keywords_path,embedding_path):

    df = pd.read_excel(keywords_path)
    # df.head(10)

    dic=defaultdict(list)
    for index,row in df.iteritems():
        # print
        dic[index].append(list(row))
    # print(dic["盗窃罪"][0])

    for key,value in dic.items():
        dic[key]=dic[key][0]
    # print(dic)

    model=Word2Vec.load(embedding_path)
    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    id2word=special_tokens+model.wv.index_to_key
    word2id=model.wv.key_to_index
    #其他对应的id往后移动特殊token总数的长度
    for k in word2id.keys():
        word2id[k]+=len(special_tokens)
    #特殊token对应id加在最前
    for i in range(len(special_tokens)):
        word2id[special_tokens[i]]=i

    for key,value in dic.items():
        sent=dic[key]
        sent=[word2id[w] if w in word2id.keys() else word2id["[UNK]"] for w in sent]
        dic[key]=sent
    # print(dic)

    return dic







