from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

import re


class BERTDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer,
                 maxlen1, maxlen2, maxlen3, pad, pair, test):
        self.pad = pad
        self.pair = pair
        transform1 = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=maxlen1, pad=self.pad, pair=self.pair)
        transform2 = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=maxlen2, pad=self.pad, pair=self.pair)
        transform3 = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=maxlen3, pad=self.pad, pair=self.pair)

        self.test = test
        def clean_text(sent):
          sent_clean=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]", " ", sent) #특수문자 및 기타 제거
          sent_clean = re.sub(' +', ' ', sent_clean) # 다중 공백 제거
          return sent_clean

        self.sentences1 = [transform1([clean_text(str(i))]) for i in dataset['과제명']]
        self.sentences2 = [transform2([clean_text(str(i))]) for i in dataset['요약문_연구목표']]
        self.sentences3 = [transform3([clean_text(str(i))]) for i in dataset['요약문_한글키워드']]
        if self.test == False:
            self.labels = [np.int32(i) for i in dataset['label']]

    def __getitem__(self, i):
        if self.test == False:
            return (self.sentences1[i] + self.sentences2[i] + self.sentences3[i] + (self.labels[i], ))
        else :
            return (self.sentences1[i] + self.sentences2[i] + self.sentences3[i])

    def __len__(self):
        return (len(self.sentences1))