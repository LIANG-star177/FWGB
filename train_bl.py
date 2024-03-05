import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from transformers import BertTokenizer, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
import os
from tokenizer import MyTokenizer
from Electra import Electra
from models.DPCNN import DPCNN
from models.TextCNN import TextCNN
from models.Bert import Bert
from models.TopJudge import TopJudge
# from textRNN import TextRNN
from data_loader import RANDOM_SEED, simple_load_confusing_data, load_law_data
from attention_visualization import createHTML
from key_words import keywords_id
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#是否clean需要改模型名、loaddata时的clean参数、最后的加载模型位置


#设置随机种子，在dataloader时shuffle
RANDOM_SEED=10
torch.manual_seed(RANDOM_SEED)

class Trainer:
    def __init__(self,load_path=None):
        # dataset_name="news"
        
        train_data_path="data/4sin_split/train.csv"
        valid_data_path="data/4sin_split/valid.csv"
        test_data_path="data/modify_test/test_tft_rob_2.csv"

        # total_data_path="data/data_train.json"
        self.batch_size=8
        self.seq_len=512
        self.epochs=100
        self.model_name="DPCNN_4sin_2"
        self.embedding_path="gensim_train/word2vec_4sin.model"
        self.keywords_path="data/key_words_4sin.xlsx"

        
        
        # self.train_set, self.valid_set, self.test_set, self.maps=load_law_data(total_data_path, self.seq_len, self.embedding_path,text_clean=False)
        self.train_set,self.maps=simple_load_confusing_data(train_data_path,self.seq_len,self.embedding_path,text_clean=False)
        self.valid_set,_=simple_load_confusing_data(valid_data_path,self.seq_len,self.embedding_path,text_clean=False)
        self.test_set,_=simple_load_confusing_data(test_data_path,self.seq_len,self.embedding_path,text_clean=False)
        
        self.keywords_dic=keywords_id(self.keywords_path,self.embedding_path)
        self.idx2charge=self.maps["idx2charge"]
        self.train_iter=DataLoader(self.train_set,batch_size=self.batch_size,shuffle=False,drop_last=True)
        self.dev_iter=DataLoader(self.valid_set,batch_size=self.batch_size,shuffle=False,drop_last=False)
        self.test_iter=DataLoader(self.test_set,batch_size=self.batch_size,shuffle=False,drop_last=False)

        #定义一个专门用于解码还原原始文字的tokenizer
        # self.tokenizer_view=MyTokenizer(self.embedding_path)
        self.tokenizer_view=BertTokenizer.from_pretrained("bert-base-chinese")
        #总的vocab数量为136352，这里要手动设置，因为前面用了三次tokenizer,
        self.model=Bert(self.tokenizer_view.vocab_size,emb_dim=256,hid_dim=256,maps=self.maps).cuda()
        # self.model=TextCNN(self.tokenizer_view.vocab_size,emb_dim=300,hid_dim=128,maps=self.maps, embedding_path=self.embedding_path).cuda()

        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=5e-5)

    def criterion(self, out, label):
        return f1_score(out.cpu().argmax(1), label.cpu(), average='micro')
    
    #将标签转换为one_hot
    def one_hot_labels(self, labels_index, maps):
        label=[0]*len(maps["charge2idx"])
        label[int(labels_index)] = 1
        return label

    def train(self):
        best_score=0        
        for epoch in range(self.epochs):
            print("-"*10+"training"+"-"*10)
            tq = tqdm(self.train_iter)
            for data in tq:
                #将data中的数据转为cuda()，
                for k in data:
                    if type(data[k]) is dict:
                        for k2 in data[k]:
                            data[k][k2] = data[k][k2].cuda()
                    else:
                        data[k] = data[k].cuda()
            
                self.optimizer.zero_grad()
                out=self.model(data)

                label=data["charge"].cuda()
                loss=nn.CrossEntropyLoss()(out["charge"],label)
                f1=self.criterion(out["charge"],label)
                loss.backward()
                #4表示保留四位小数，detach()将loss从计算图里抽离出来
                tq.set_postfix(epoch=epoch, train_loss=np.around(loss.cpu().detach().numpy(),4), train_f1=f1)                
                self.optimizer.step()

            #一个epoch输出一次validation的结果
            print("-"*10+"validating"+"-"*10)
            dev_out=[]
            loss_sum=0.0
            cnt=0
            tq = tqdm(self.dev_iter)
            for data in tq:
                cnt+=1
                for k in data:
                    if type(data[k]) is dict:
                        for k2 in data[k]:
                            data[k][k2] = data[k][k2].cuda()
                    else:
                        data[k] = data[k].cuda()
                with torch.no_grad():
                    out=self.model(data)
                    # attention = attention.squeeze(axis=2).data.numpy()
                label=data["charge"].cuda()
                loss=nn.CrossEntropyLoss()(out["charge"],label)
                loss_sum+=np.around(loss.cpu().detach().numpy(),4)
                f1=self.criterion(out["charge"],label)
                tq.set_postfix(epoch=epoch,dev_loss=np.around(loss.cpu().detach().numpy(),4),dev_f1=f1)
                dev_out.append((out["charge"].cpu().argmax(dim=1),label.cpu()))
            loss_batch=loss_sum/cnt
            pred=torch.cat([i[0] for i in dev_out])
            truth=torch.cat([i[1] for i in dev_out])
            valid_micro_f1=f1_score(truth, pred, average='micro')
            valid_macro_f1=f1_score( truth, pred, average='macro')
            print("valid micro f1:", valid_micro_f1)
            print("valid macro f1:", valid_macro_f1)
            print("valid_loss:", loss_batch)

            #在性能提升的情况下保存当前epoch模型,将每个epoch的validation结果输入
            model_save_path="logs/{}/".format(self.model_name)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            save_path=model_save_path+"model_{}".format(epoch)

            with open("logs/{}/validation2.txt".format(self.model_name),"a") as f:
                f.write(str(epoch)+" epoch"+"\n")
                f.write("valid_loss: "+str(loss_batch)+"\n")

            if valid_micro_f1>best_score:
                best_score=valid_micro_f1
                best_model_path=save_path
                torch.save(self.model,best_model_path)
        
        self.evaluate(best_model_path)

    def evaluate(self,save_path):
        truth_out=[[] for _ in range(4)]
        pred_out=[[] for _ in range(4)]
        print("-"*10+"testing"+"-"*10)
        test_out=[]
        model=torch.load(save_path)
        bat_cnt=0
        for data in tqdm(self.test_iter):
            bat_cnt+=1
            for k in data:
                if type(data[k]) is dict:
                    for k2 in data[k]:
                        data[k][k2] = data[k][k2].cuda()
                else:
                    data[k] = data[k].cuda()
            with torch.no_grad():
                out=model(data)

            label=data["charge"].cuda()


            test_out.append((out["charge"].cpu().argmax(dim=1),label.cpu()))
            for i in range(len(label)):
                pred_out[label[i]].append(self.idx2charge[str(out["charge"].argmax(dim=1)[i].item())])
                truth_out[label[i]].append(self.idx2charge[str(label[i].item())])

        pred=torch.cat([i[0] for i in test_out])
        truth=torch.cat([i[1] for i in test_out])
        print(precision_recall_fscore_support(truth,pred))
        valid_micro_f1=f1_score(truth, pred, average='micro')
        valid_macro_f1=f1_score(truth, pred, average='macro')
        print("test micro f1:", valid_micro_f1)
        print("test macro f1:", valid_macro_f1)
        print(confusion_matrix(pred,truth))


    def evaluate2(self,save_path):
        truth_out=[[] for _ in range(4)]
        pred_out=[[] for _ in range(4)]
        print("-"*10+"validating"+"-"*10)
        valid_out=[]
        model=torch.load(save_path)
        bat_cnt=0
        for data in tqdm(self.dev_iter):
            bat_cnt+=1
            for k in data:
                if type(data[k]) is dict:
                    for k2 in data[k]:
                        data[k][k2] = data[k][k2].cuda()
                else:
                    data[k] = data[k].cuda()
            with torch.no_grad():
                out=model(data)
            label=data["charge"].cuda()
            valid_out.append((out["charge"].cpu().argmax(dim=1),label.cpu()))
            for i in range(len(label)):
                pred_out[label[i]].append(self.idx2charge[str(out["charge"].argmax(dim=1)[i].item())])
                truth_out[label[i]].append(self.idx2charge[str(label[i].item())])

        pred_valid=torch.cat([i[0] for i in valid_out]).tolist()
        truth_valid=torch.cat([i[1] for i in valid_out]).tolist()

        truth_out=[[] for _ in range(4)]
        pred_out=[[] for _ in range(4)]
        print("-"*10+"testing"+"-"*10)
        test_out=[]
        model=torch.load(save_path)
        bat_cnt=0
        for data in tqdm(self.test_iter):
            bat_cnt+=1
            for k in data:
                if type(data[k]) is dict:
                    for k2 in data[k]:
                        data[k][k2] = data[k][k2].cuda()
                else:
                    data[k] = data[k].cuda()
            with torch.no_grad():
                out,attention=model(data)
            label=data["charge"].cuda()
            test_out.append((out["charge"].cpu().argmax(dim=1),label.cpu()))
            for i in range(len(label)):
                pred_out[label[i]].append(self.idx2charge[str(out["charge"].argmax(dim=1)[i].item())])
                truth_out[label[i]].append(self.idx2charge[str(label[i].item())])

        pred_test=torch.cat([i[0] for i in test_out]).tolist()
        truth_test=torch.cat([i[1] for i in test_out]).tolist()

        pred_valid.extend(pred_test)
        truth_valid.extend(truth_test)

        print(precision_recall_fscore_support(truth_valid,pred_valid))


if __name__=="__main__":
    trainer= Trainer()
    # trainer.train()
    trainer.evaluate("logs/bert_4sin/model_2")
 
