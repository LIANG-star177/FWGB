import pickle
from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
import random
import os
import jieba
import re
import json
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer
import torch.nn as nn
from tokenizer import MyTokenizer

RANDOM_SEED = 16
torch.manual_seed(RANDOM_SEED)

def text_cleaner(text):
    def load_stopwords(filename):
        stopwords = []
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.replace("\n", "")
                stopwords.append(line)
        return stopwords

    stop_words = load_stopwords("data/stopword.txt")

    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        # newline after </p> and </div> and <h1/>...
        {r'</(div)\s*>\s*': u'\n'},
        # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        # show links instead of texts
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]

    # 替换html特殊字符
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace(
        "&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    text = text.replace('+', ' ').replace(',', ' ').replace(':', ' ')
    text = re.sub("([0-9]+[年月日])+", "", text)
    text = re.sub("[a-zA-Z]+", "", text)
    text = re.sub("[0-9\.]+元", "", text)
    stop_words_user = ["年", "月", "日", "时", "分", "许", "某", "甲", "乙", "丙"]
    word_tokens = jieba.cut(text)

    def str_find_list(string, words):
        for word in words:
            if string.find(word) != -1:
                return True
        return False

    text = [w for w in word_tokens if w not in stop_words if not str_find_list(w, stop_words_user)
            if len(w) >= 1 if not w.isspace()]
    return " ".join(text)

#只清洗数据，不去停用词。
def text_cleaner2(text):
    # 替换html特殊字符
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace(
        "&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    # 换行替换为#, 空格替换为&
    text = text.replace("#", "").replace("$", "").replace("&", "")
    text = text.replace("\n", "").replace(" ", "")

    return text

#用的时候改改属性即可，中文的需要bert来tokenizer
class myDataset(Dataset):
    def __init__(self, justice,  charge):
        self.justice = justice
        self.charge= torch.LongTensor(charge)

    def __getitem__(self, idx):
        #为了匹配bert_tokenizer的返回结果
        return {"justice":
                {
                    "input_ids": self.justice["input_ids"][idx],
                    "token_type_ids": self.justice["token_type_ids"][idx],
                    "attention_mask": self.justice["attention_mask"][idx],
                }, "charge": self.charge[idx]}

    def __len__(self):
        return len(self.charge)

def get_split_dataset(idx,justice,charge):
    #此处根据ID返回对应的数据，因为dataset需要根据id进行划分
    justice_cur = {
        "input_ids": justice["input_ids"][idx],
        "token_type_ids": justice["token_type_ids"][idx],
        "attention_mask": justice["attention_mask"][idx],
    }
    charge_cur=pd.Series(charge)[idx].tolist()
    return myDataset(justice_cur,charge_cur)

def simple_load_data(data_path,seq_len,clean:bool):
    data = pd.read_csv(data_path,sep="\t",encoding='utf-8',header=None,names=["fact","type"])

    #建一个存clean后的dataset的文件夹并保存clean文件
    path, file = os.path.split(data_path)
    stem, suffix = os.path.splitext(file)
    # print(path,file,stem,suffix)

    if clean:
        file = stem+"_clean"+suffix
        clean_data_path = os.path.join(path, file)
        if not os.path.exists(clean_data_path):
            for i in tqdm(range(len(data))):
                data["fact"][i] = text_cleaner(data["fact"][i])
            data.to_csv(clean_data_path, index=False, sep=",")
        data = pd.read_csv(clean_data_path,sep=",")
    # print(data.head(10))

    #清除数据中含有None的行
    if len(data)!=len(data.dropna()):
        print("before dropna, data num:",len(data))
        print("after dropna, data num:",len(data.dropna()))
    data=data.dropna()
    data=data.reset_index()
    # print(data.head(10))

    #tokenizer
    #根据clean参数看保存
    pkl_path = "data/pkl/"+stem+"_clean.pkl" if clean else "data/pkl/"+stem+".pkl"
    # print(data["fact"].tolist())
    if not os.path.exists(pkl_path):
        tokenizer=BertTokenizer.from_pretrained("bert-base-chinese")
        fact=data["fact"].tolist()
        fact=tokenizer(fact,return_tensors="pt",padding="max_length",max_length=seq_len,truncation=True)
        with open(pkl_path,"wb") as f:
            pickle.dump(fact,f,protocol=pickle.HIGHEST_PROTOCOL)
        print("pkl saved: {}".format(pkl_path))
    
    else:
        with open(pkl_path,"rb") as f:
            fact=pickle.load(f)
    # print(fact)
    type=data["type"].tolist()
    # print(type)

    #根据idx创建dataset
    idx=list(range(len(type)))
    # print(idx)
    # print(fact)
    # print(type)
    dataset=get_split_dataset(idx,fact,type)
    return dataset


#此时只用了justice预测judge
def load_confusing_data(filename, seq_len, embedding_path, train_set_ratio=0.8, text_clean=True):
    #LA:将错误行跳过
    df = pd.read_csv(filename, sep=",",encoding="gbk")
    path, file = os.path.split(filename)
    stem, suffix = os.path.splitext(file)

    # 读入csv数据，根据上面text_clean参数是否过stopwords
    if text_clean:
        file = stem+"_clean"+suffix
        clean_data_path = os.path.join(path, file)
        if not os.path.exists(clean_data_path):
            for i in tqdm(range(len(df))):
                # df["charge"][i] = text_cleaner(df["charge"][i])
                df["justice"][i] = text_cleaner(df["justice"][i])
                df["opinion"][i] = text_cleaner(df["opinion"][i])
            df.to_csv(clean_data_path, index=False, sep=",")
        df = pd.read_csv(clean_data_path)

    if len(df) != len(df.dropna()):
        print("before drop nan, data num: ", len(df))
        print("after drop nan, data num: ", len(df.dropna()))
    df = df.dropna()
    df = df.reset_index()

    # 将fact和opinion文本进行tokenize
    pkl_path = "data/pkl/"+stem+"_clean.pkl" if text_clean else "data/pkl/"+stem+".pkl"

    if not os.path.exists(pkl_path):
        #用字划分数据集得到id，使用现有的Bert_tokenizer
        # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        #用词划分，使用自己的tokenizer,替换时注意删除原有的pkl文件
        tokenizer=MyTokenizer(embedding_path)

        justice = df["justice"].tolist()
        opinion = df["opinion"].tolist()
        justice = tokenizer(justice, return_tensors="pt",
                         padding="max_length", max_length=seq_len, truncation=True)
        opinion = tokenizer(opinion, return_tensors="pt",
                            padding="max_length", max_length=seq_len, truncation=True)
        with open(pkl_path, "wb") as f:
            pickle.dump((justice, opinion), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    with open(pkl_path, "rb") as f:
        justice, opinion = pickle.load(f)

    charge = df["charge"].tolist()
    # judge = df["judge"].tolist() if "judge" in df.columns else [-1]*len(charge)
    # article = df["article"].tolist() if "article" in df.columns else [-1]*len(charge)

    # use_violence = df['use_violence'].tolist()
    # violence_target = df['violence_target'].tolist()
    # violence_victim = df['violence_victim'].tolist()
    # violate_victim = df['violate_victim'].tolist()
    # use_contract = df['use_contract'].tolist()

    # 标签转数字id

    def label2idx(label):
        st = set(label)
        lst = sorted(list(st))  # 按照字符串顺序排列
        mp_label2idx, mp_idx2label = dict(), dict()
        for i in range(len(lst)):
            mp_label2idx[lst[i]] = i
            mp_idx2label[i] = lst[i]
        return [mp_label2idx[i] for i in label], mp_label2idx, mp_idx2label
    charge, mp_charge2idx, mp_idx2charge = label2idx(charge)
    # article, mp_article2idx, mp_idx2article = label2idx(article)

    maps = {}
    maps["charge2idx"] = mp_charge2idx
    maps["idx2charge"] = mp_idx2charge
    # maps["article2idx"] = mp_article2idx
    # maps["idx2article"] = mp_idx2article

    # 划分trainset, validset
    data_split_dir = "data/data_split/20211216/"
    if not os.path.exists(data_split_dir):
        os.mkdir(data_split_dir)

    

    tot_size = len(df)
    train_size = int(tot_size*train_set_ratio)
    valid_size = int((tot_size-train_size)/2)
    test_size = tot_size-train_size-valid_size
    random.seed(RANDOM_SEED)
    shuffle_idx = list(range(len(charge)))
    random.shuffle(shuffle_idx)
    train_idx, valid_idx, test_idx = shuffle_idx[:train_size], shuffle_idx[
        train_size:train_size+valid_size], shuffle_idx[train_size+valid_size:]
    train_df, valid_df, test_df = df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]
    train_df.to_csv(data_split_dir+"train.csv", index=False, sep=",")
    valid_df.to_csv(data_split_dir+"valid.csv", index=False, sep=",")
    test_df.to_csv(data_split_dir+"test.csv", index=False, sep=",")

    trainset = get_split_dataset(
        train_idx, justice, charge)
    validset = get_split_dataset(
        valid_idx, justice, charge)
    testset = get_split_dataset(
        test_idx, justice, charge)

    return trainset, validset, testset, maps

def simple_load_confusing_data(filename, seq_len, embedding_path, text_clean:bool):
    #LA:将错误行跳过
    df = pd.read_csv(filename, sep=",")
    path, file = os.path.split(filename)
    stem, suffix = os.path.splitext(file)

    # 读入csv数据，根据上面text_clean参数是否过stopwords
    if text_clean:
        file = stem+"_clean"+suffix
        clean_data_path = os.path.join(path, file)
        if not os.path.exists(clean_data_path):
            for i in tqdm(range(len(df))):
                # df["charge"][i] = text_cleaner(df["charge"][i])
                df["justice"][i] = text_cleaner2(df["justice"][i])
                # df["opinion"][i] = text_cleaner2(df["opinion"][i])
            df.to_csv(clean_data_path, index=False, sep=",")
        df = pd.read_csv(clean_data_path)

    if len(df) != len(df.dropna()):
        print("before drop nan, data num: ", len(df))
        print("after drop nan, data num: ", len(df.dropna()))
    df = df.dropna()
    df = df.reset_index()

    # 将fact和opinion文本进行tokenize
    # pkl_path = "data/pkl_4sin/"+stem+"_clean.pkl" if text_clean else "data/pkl_4sin/"+stem+".pkl"
    pkl_path = "data/pkl_4sin_el/"+stem+"_clean.pkl" if text_clean else "data/pkl_4sin_el/"+stem+".pkl"
    if not os.path.exists(pkl_path):
        #用字划分数据集得到id，使用现有的Bert_tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # 用词划分，使用自己的tokenizer,替换时注意删除原有的pkl文件
        # tokenizer=MyTokenizer(embedding_path)

        justice = df["justice"].tolist()
        # opinion = df["opinion"].tolist()
        justice = tokenizer(justice, return_tensors="pt",
                         padding="max_length", max_length=seq_len, truncation=True)
        # opinion = tokenizer(opinion, return_tensors="pt",
        #                     padding="max_length", max_length=seq_len, truncation=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(justice, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pkl_path, "rb") as f:
            justice= pickle.load(f)

    charge = df["charge"].tolist()

    # 标签转数字id
    def label2idx(label):
        maps = {}
        c2i_path = "data/charge2idx.json"
        with open(c2i_path) as f:
            c2i = json.load(f)
            maps["charge2idx"] = c2i
            maps["idx2charge"] = {str(v): k for k, v in c2i.items()}
        return [maps["charge2idx"][i] for i in label], maps["charge2idx"], maps["idx2charge"]
    charge, mp_charge2idx, mp_idx2charge = label2idx(charge)
    # article, mp_article2idx, mp_idx2article = label2idx(article)

    maps = {}
    maps["charge2idx"] = mp_charge2idx
    maps["idx2charge"] = mp_idx2charge
    # maps["article2idx"] = mp_article2idx
    # maps["idx2article"] = mp_idx2article
    idx=list(range(len(charge)))
    dataset = get_split_dataset(idx, justice, charge)

    return dataset, maps

def load_cail2018_data(filename, seq_len, embedding_path, text_clean:bool):
    #LA:将错误行跳过
    source=[]
    with open(filename,'r',encoding="utf-8") as f:
        for line in f:
            source.append((json.loads(line)))
    df_data=pd.DataFrame(source)
    charge=[]
    justice=[]
    for i in range(len(df_data)):
        # print(df_data["meta"].iloc[i]["accusation"])
        if len(df_data["meta"][i]["accusation"])==1:
            tmp=str(df_data["meta"][i]["accusation"])
            tmp=tmp.replace("[","").replace("]","").replace("'","")
            charge.append(tmp+"罪")
            justice.append(df_data["fact"][i])
    df=pd.DataFrame()
    df["charge"]=charge
    df["justice"]=justice
    path, file = os.path.split(filename)
    stem, suffix = os.path.splitext(file)

    # 读入csv数据，根据上面text_clean参数是否过stopwords
    if text_clean:
        file = stem+"_clean"+suffix
        clean_data_path = os.path.join(path, file)
        if not os.path.exists(clean_data_path):
            for i in tqdm(range(len(df))):
                # df["charge"][i] = text_cleaner(df["charge"][i])
                df["justice"][i] = text_cleaner2(df["justice"][i])
                # df["opinion"][i] = text_cleaner2(df["opinion"][i])
            df.to_csv(clean_data_path, index=False, sep=",")
        df = pd.read_csv(clean_data_path)

    if len(df) != len(df.dropna()):
        print("before drop nan, data num: ", len(df))
        print("after drop nan, data num: ", len(df.dropna()))
    df = df.dropna()
    df = df.reset_index()

    # 将fact和opinion文本进行tokenize
    pkl_path = "data/pkl18_new/"+stem+"_clean.pkl" if text_clean else "data/pkl18_new/"+stem+".pkl"
    if not os.path.exists(pkl_path):
        #用字划分数据集得到id，使用现有的Bert_tokenizer
        # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        #用词划分，使用自己的tokenizer,替换时注意删除原有的pkl文件
        tokenizer=MyTokenizer(embedding_path)

        justice = df["justice"].tolist()
        # opinion = df["opinion"].tolist()
        justice = tokenizer(justice, return_tensors="pt",
                         padding="max_length", max_length=seq_len, truncation=True)
        # opinion = tokenizer(opinion, return_tensors="pt",
        #                     padding="max_length", max_length=seq_len, truncation=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(justice, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pkl_path, "rb") as f:
            justice= pickle.load(f)

    charge = df["charge"].tolist()

    # 标签转数字id
    def label2idx(label):
        st = set(label)
        lst = sorted(list(st))  # 按照字符串顺序排列
        mp_label2idx, mp_idx2label = dict(), dict()
        for i in range(len(lst)):
            mp_label2idx[lst[i]] = i
            mp_idx2label[i] = lst[i]
        return [mp_label2idx[i] for i in label], mp_label2idx, mp_idx2label
    charge, mp_charge2idx, mp_idx2charge = label2idx(charge)
    # article, mp_article2idx, mp_idx2article = label2idx(article)

    maps = {}
    maps["charge2idx"] = mp_charge2idx
    maps["idx2charge"] = mp_idx2charge

     # 划分trainset, validset
    data_split_dir = "data/cail2018train_split/"
    if not os.path.exists(data_split_dir):
        os.mkdir(data_split_dir)

    train_set_ratio=0.8
    tot_size = len(df)
    train_size = int(tot_size*train_set_ratio)
    valid_size = int((tot_size-train_size)/2)
    test_size = tot_size-train_size-valid_size
    random.seed(RANDOM_SEED)
    shuffle_idx = list(range(len(charge)))
    random.shuffle(shuffle_idx)
    train_idx, valid_idx, test_idx = shuffle_idx[:train_size], shuffle_idx[
        train_size:train_size+valid_size], shuffle_idx[train_size+valid_size:]
    train_df, valid_df, test_df = df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]
    train_df.to_csv(data_split_dir+"train.csv", index=False, sep=",")
    valid_df.to_csv(data_split_dir+"valid.csv", index=False, sep=",")
    test_df.to_csv(data_split_dir+"test.csv", index=False, sep=",")

    train_set = get_split_dataset(train_idx, justice, charge)
    valid_set = get_split_dataset(valid_idx, justice, charge)
    test_set = get_split_dataset(test_idx, justice, charge)

    return train_set, valid_set, test_set, maps

def load_law_data(filename, seq_len, embedding_path, text_clean:bool):
    #LA:将错误行跳过
    df=pd.read_csv(filename, sep=",")
    path, file = os.path.split(filename)
    stem, suffix = os.path.splitext(file)

    # 读入csv数据，根据上面text_clean参数是否过stopwords
    if text_clean:
        file = stem+"_clean"+suffix
        clean_data_path = os.path.join(path, file)
        if not os.path.exists(clean_data_path):
            for i in tqdm(range(len(df))):
                # df["charge"][i] = text_cleaner(df["charge"][i])
                df["justice"][i] = text_cleaner2(df["justice"][i])
                # df["opinion"][i] = text_cleaner2(df["opinion"][i])
            df.to_csv(clean_data_path, index=False, sep=",")
        df = pd.read_csv(clean_data_path)

    if len(df) != len(df.dropna()):
        print("before drop nan, data num: ", len(df))
        print("after drop nan, data num: ", len(df.dropna()))
    df = df.dropna()
    df = df.reset_index()

    # 将fact和opinion文本进行tokenize
    pkl_path = "data/pkl_6sin/"+stem+"_clean.pkl" if text_clean else "data/pkl_6sin/"+stem+".pkl"
    if not os.path.exists(pkl_path):
        #用字划分数据集得到id，使用现有的Bert_tokenizer
        # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        #用词划分，使用自己的tokenizer,替换时注意删除原有的pkl文件
        tokenizer=MyTokenizer(embedding_path)

        justice = df["justice"].tolist()
        # opinion = df["opinion"].tolist()
        justice = tokenizer(justice, return_tensors="pt",
                         padding="max_length", max_length=seq_len, truncation=True)
        # opinion = tokenizer(opinion, return_tensors="pt",
        #                     padding="max_length", max_length=seq_len, truncation=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(justice, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pkl_path, "rb") as f:
            justice= pickle.load(f)

    charge = df["charge"].tolist()

    # 标签转数字id
    def label2idx(label):
        st = set(label)
        lst = sorted(list(st))  # 按照字符串顺序排列
        mp_label2idx, mp_idx2label = dict(), dict()
        for i in range(len(lst)):
            mp_label2idx[lst[i]] = i
            mp_idx2label[i] = lst[i]
        return [mp_label2idx[i] for i in label], mp_label2idx, mp_idx2label
    charge, mp_charge2idx, mp_idx2charge = label2idx(charge)
    # article, mp_article2idx, mp_idx2article = label2idx(article)

    maps = {}
    maps["charge2idx"] = mp_charge2idx
    maps["idx2charge"] = mp_idx2charge

     # 划分trainset, validset
    data_split_dir = "data/6sin_split/"
    if not os.path.exists(data_split_dir):
        os.mkdir(data_split_dir)
    RANDOM_SEED=22
    random.seed(RANDOM_SEED)
    shuffle_idx = list(range(len(df)))
    random.shuffle(shuffle_idx)
    print(df.shape)

    train_set_ratio=0.8
    tot_size = len(df)
    train_size = int(tot_size*train_set_ratio)
    valid_size = int((tot_size-train_size)/2)

    train_idx = shuffle_idx[:train_size]
    valid_idx = shuffle_idx[train_size:train_size+valid_size]
    test_idx=shuffle_idx[train_size+valid_size:]

    train_df, valid_df, test_df = df.iloc[train_idx], df.iloc[valid_idx],df.iloc[test_idx]

    train_df.to_csv(data_split_dir+"train.csv", index=False, sep=",")
    valid_df.to_csv(data_split_dir+"valid.csv", index=False, sep=",")
    test_df.to_csv(data_split_dir+"test.csv", index=False, sep=",")

    train_set = get_split_dataset(train_idx, justice, charge)
    valid_set = get_split_dataset(valid_idx, justice, charge)
    test_set = get_split_dataset(test_idx, justice, charge)

    return train_set, valid_set, test_set, maps

#测试dataloader
if __name__=="__main__":
    total_data_path="/mnt/data/wuyiquan/liang/charge_prediction/data/total_6sin.csv"
    train_set, valid_set, test_set, maps=load_law_data(total_data_path,seq_len=512,embedding_path="/mnt/data/wuyiquan/liang/charge_prediction/gensim_train/word2vec_6sin.model",text_clean=False)

    train_iter=DataLoader(train_set,batch_size=128,shuffle=False,drop_last=True)
    valid_iter=DataLoader(valid_set,batch_size=128,shuffle=False,drop_last=False)
    test_iter=DataLoader(test_set,batch_size=128,shuffle=False,drop_last=False)

    #测试dataiter
    cnt=0
    for data in enumerate(valid_iter):
        print(data)
        cnt+=1
        if cnt>1:
            break




