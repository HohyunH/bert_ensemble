import pandas as pd
import re
import random
from kobert.utils import get_tokenizer
from tqdm import tqdm
import gluonnlp as nlp
from kobert.pytorch_kobert import get_pytorch_kobert_model
from imblearn.under_sampling import *
from imblearn.combine import *
import pickle


def preprocessing(path):
    train = pd.read_csv(f'{path}/train.csv')
    test = pd.read_csv(f'{path}/test.csv')
    sample_submission = pd.read_csv(f'{path}/sample_submission.csv')

    train['data'] = train['사업명'] + train['과제명'] + train['요약문_연구내용']  # +train['요약문_기대효과']
    test['data'] = test['사업명'] + test['과제명'] + test['요약문_연구내용']  # +test['요약문_기대효과']

    binary = []
    for lbls in train['label']:
        if lbls == 0:
            binary.append(0)
        else:
            binary.append(1)
    train['binary'] = binary

    dataset_train = train[['data', 'label', 'binary']]
    dataset_test = test[['data']]
    return dataset_train, dataset_test, sample_submission


def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-하-ㅣ]", " ", sent)
    return sent_clean


def transforming(max_len=256, pad=True, pair=False):
    random.seed(777)
    tokenizer = get_tokenizer()
    bert_model, vocab = get_pytorch_kobert_model()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=max_len, pad=pad, pair=pair)
    return bert_model, tok, transform


def tokenizing(transform, dataset_train):
    token_vectors = [transform([clean_text(str(i))])[0] for i in tqdm(dataset_train['data'])]
    return token_vectors


def ENN(token_vectors, dataset_train):
    sampler = EditedNearestNeighbours()
    X_res, y_res = sampler.fit_resample(token_vectors, dataset_train['binary'])
    return X_res, y_res


def SMOTEENN(token_vectors, dataset_train):
    sampler = SMOTEENN(random_state=777)
    X_res, y_res = sampler.fit_resample(token_vectors, dataset_train['label'])
    return X_res, y_res


if __name__ == "__main__":
    # dataset_train, dataset_test, sample_submission = preprocessing('./enn_bert/data')
    # bert_model, tok, transform = transforming()
    # token_vectors = tokenizing(transform, dataset_train)
    # X_res, y_res = ENN(token_vectors, dataset_train)

    with open('ENN_X_128.pickle', 'rb') as file:
        X_res = pickle.load(file)
    with open('ENN_y_128.pickle', 'rb') as file:
        y_res = pickle.load(file)

    print(X_res)
    print(len(X_res))
    print(len(X_res[0]))