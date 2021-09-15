# import konlpy
# from konlpy.tag import Okt
import random
random.seed(777)
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import gluonnlp as nlp

from tqdm import tqdm

from model import BERTClassifier

from dataset import BERTDataset

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]", " ", str(sent))  # 특수문자 및 기타 제거
    sent_clean = re.sub(' +', ' ', sent_clean)  # 다중 공백 제거
    return sent_clean


def data_preprocessing(data, target_columns):
    data = data.fillna('NONE')
    data['요약문_연구내용'] = data.apply(lambda x: x['과제명'] if x['요약문_연구내용'] == 'NONE' else x['요약문_연구내용'], axis=1)

    # okt = Okt()
    data['요약문_기대효과'] = data.apply(lambda x: x['과제명'] if x['요약문_기대효과'] == 'NONE' else x['요약문_기대효과'], axis=1)

    data.loc[:, target_columns] = data[target_columns].applymap(lambda x: clean_text(x))

    return data


def drop_short_texts(train, target_columns):
    train_index = set(train.index)
    for column in target_columns:
        train_index -= set(train[train[column].str.len() < 10].index)

    train = train.loc[list(train_index)]

    train.loc[:, target_columns] = train[target_columns].applymap(lambda x: clean_text(x))

    print('SHORT TEXTS DROPED')

    return train


def sampling_data(train, target_columns):
    pj_name_len = 18
    summ_goal_len = 210
    summ_key_len = 120

    max_lens = [pj_name_len, summ_goal_len, summ_key_len]
    total_index = set(train.index)
    for column, max_len in zip(target_columns, max_lens):
        temp = train[column].apply(lambda x: len(x.split()) < max_len)
        explained_ratio = temp.values.sum() / train.shape[0]
        # display(column)
        # display(f'전체 샘플 중 길이가 {max_len}이하인 샘플의 비율 : {round(explained_ratio * 100, 2)}%')

        total_index -= set(train[temp == False].index)
    train = train.loc[list(total_index)].reset_index(drop=True)

    return train

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def evaluate(model, eval_loader, optimizer, device):
    model.eval()
    train_acc = 0.0
    with torch.no_grad():
      for batch_id, (token_ids, valid_length, segment_ids, token_ids2, valid_length2, segment_ids2, token_ids3, valid_length3, segment_ids3, label) in enumerate(tqdm(eval_loader)):
          optimizer.zero_grad()
          token_ids = token_ids.long().to(device)
          segment_ids = segment_ids.long().to(device)
          valid_length= valid_length
          token_ids2 = token_ids2.long().to(device)
          segment_ids2 = segment_ids2.long().to(device)
          valid_length2= valid_length2
          token_ids3 = token_ids3.long().to(device)
          segment_ids3 = segment_ids3.long().to(device)
          valid_length3= valid_length3
          label = label.long().to(device)
          out = model(token_ids, valid_length, segment_ids, token_ids2, valid_length2, segment_ids2, token_ids3, valid_length3, segment_ids3)
          train_acc += calc_accuracy(out, label)
    print("ACC : ", train_acc / (batch_id+1))



def oversampling_minor_classes(train, target_columns):
    temp = train.copy()
    temp.loc[:, target_columns] = temp.loc[:, target_columns].applymap(lambda x: len(x.split()))
    pj_range = (6, 11)
    summ_goal_range = (30, 89)
    summ_key_range = (5, 9)
    temp = temp.query('label != 0')
    temp = pd.DataFrame(list(train.loc[temp.query(
        '과제명 in @pj_range or 요약문_연구목표 in @summ_goal_range or 요약문_한글키워드 in @summ_key_range').index].values) * 1,
                        columns=temp.columns)
    train = pd.concat([train, temp], axis=0).reset_index(drop=True)
    train.groupby(['label'])['과제명'].agg('count').plot.bar()
    plt.show()
    train.groupby(['label'])['과제명'].agg('count')[1:].plot.bar()
    plt.show()

    return train

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=5, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.logits = logits
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss) # prevents nanas when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def main(args):
    device = torch.device("cuda")

    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    sample_submission = pd.read_csv('./sample_submission.csv')

    target_columns = ['과제명', '요약문_연구내용', '요약문_기대효과']
    # train_texts = train.copy()
    train_texts = data_preprocessing(train, target_columns)
    print('TRAIN NA DATA NUM : ', train_texts.isna().sum().sum())
    train_texts = drop_short_texts(train_texts, target_columns)
    train_texts = sampling_data(train_texts, target_columns)
    train_texts = oversampling_minor_classes(train_texts, target_columns)

    train_dataset, val_dataset = train_texts[:100000], train_texts[100000:]

    test_texts = data_preprocessing(test, target_columns)
    # test_texts = test.copy()
    print('TEST NA DATA NUM : ', train_texts.isna().sum().sum())

    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    batch_size = args.batch_size
    warmup_ratio = 0.1
    num_epochs = args.epochs
    max_grad_norm = 1
    log_interval = 200
    learning_rate = 5e-5

    data_train = BERTDataset(train_dataset, tok, args.maxlen1, args.maxlen2, args.maxlen3, True, False, False)
    data_val = BERTDataset(val_dataset, tok, args.maxlen1, args.maxlen2, args.maxlen3, True, False, False)
    data_test = BERTDataset(test_texts, tok, args.maxlen1, args.maxlen2, args.maxlen3, True, False, True)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5, shuffle=False)

    model = BERTClassifier(bertmodel, 46, dr_rate=0.5).to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    if args.loss == "CE":
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
    else :
        loss_fn = FocalLoss()

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    model.load_state_dict(torch.load('./bert_model.pt'))
    # model.train()
    # for e in range(num_epochs):
    #     train_acc = 0.0
    #     test_acc = 0.0
    #     for batch_id, (token_ids, valid_length, segment_ids, token_ids2, valid_length2, segment_ids2, token_ids3, valid_length3, segment_ids3, label) in enumerate(tqdm(train_dataloader)):
    #         optimizer.zero_grad()
    #         token_ids = token_ids.long().to(device)
    #         segment_ids = segment_ids.long().to(device)
    #         valid_length = valid_length
    #         token_ids2 = token_ids2.long().to(device)
    #         segment_ids2 = segment_ids2.long().to(device)
    #         valid_length2 = valid_length2
    #         token_ids3 = token_ids3.long().to(device)
    #         segment_ids3 = segment_ids3.long().to(device)
    #         valid_length3 = valid_length3
    #         label = label.long().to(device)
    #         out = model(token_ids, valid_length, segment_ids, token_ids2, valid_length2, segment_ids2, token_ids3,
    #                     valid_length3, segment_ids3)
    #         loss = loss_fn(out, label)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    #         optimizer.step()
    #         scheduler.step()  # Update learning rate schedule
    #         train_acc += calc_accuracy(out, label)
    #         if batch_id % log_interval == 0:
    #             print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
    #                                                                      train_acc / (batch_id + 1)))
    #             torch.save(model.state_dict(), "./bert_model.pt")
    #
    #     print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
    evaluate(model, val_dataloader, optimizer, device)

    answers = []
    model.eval()
    with torch.no_grad():
        for batch_id, (
                token_ids, valid_length, segment_ids, token_ids2, valid_length2, segment_ids2, token_ids3,
                valid_length3,
                segment_ids3) in enumerate(tqdm(test_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            token_ids2 = token_ids2.long().to(device)
            segment_ids2 = segment_ids2.long().to(device)
            valid_length2 = valid_length2
            token_ids3 = token_ids3.long().to(device)
            segment_ids3 = segment_ids3.long().to(device)
            valid_length3 = valid_length3
            out = model(token_ids, valid_length, segment_ids, token_ids2, valid_length2, segment_ids2, token_ids3,
                        valid_length3, segment_ids3)
            answers.extend(torch.max(out, 1)[1].tolist())

    sample_submission['label'] = answers

    sample_submission.to_csv('./ensemble_bert_baseline.csv', index=False)

    print("FINISH!!")

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import argparse

    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    parser.add_argument('--maxlen1', '-ml1', type=int, default=32, help='Max length')
    parser.add_argument('--maxlen2', '-ml2', type=int, default=256, help='Max length')
    parser.add_argument('--maxlen3', '-ml3', type=int, default=32, help='Max length')
    parser.add_argument('--batch-size', '-bs', type=int, default=24, help='Batch size')
    parser.add_argument('--epochs', '-ep', type=int, default=10, help='Number of epochs')
    parser.add_argument('--loss', '-ls', type=str, default='CE', help='Name of loss function')
    args = parser.parse_args()

    main(args)
