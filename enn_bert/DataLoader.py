from torch.utils.data import Dataset
import gluonnlp as nlp
import re
import numpy as np


class BERTDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, max_len,
                 pad, pair, test):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.test = test

        def clean_text(sent):
            sent_clean = re.sub("[^가-힣ㄱ-하-ㅣ]", " ", sent)
            return sent_clean

        self.sentences = [transform([clean_text(str(i))]) for i in dataset['data']]
        if not self.test:
            self.labels = [np.int32(i) for i in dataset['label']]
            self.binary = [np.int32(i) for i in dataset['binary']]

    def __getitem__(self, i):
        if not self.test:
            return (self.sentences[i] + (self.labels[i],)), self.binary[i]
        else:
            return self.sentences[i]

    def __len__(self):
        return len(self.sentences)
