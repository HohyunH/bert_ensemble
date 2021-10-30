# Dacon BERT Classification

본 코드는 [DACON 자연어 기반 기후기술분류 AI 경진대회] 공모전 실험 코드 입니다.

"국가 연구개발과제를 '기후기술분류체계'에 맞추어 라벨링하는 알고리즘 개발" 의 주제로 진행된 대회 입니다.

두 가지 방법으로 결과를 도출하였습니다.

## 1. BERT Ensemble

- BERT 모델 3개를 앙상블하여 결과를 낸 방식으로 각각 다른 Column의 데이터를 입력시켜서 합치는 식으로 모델을 구성했다.

```python
    def forward(self, token_ids, valid_length, segment_ids, token_ids2, valid_length2, segment_ids2, token_ids3,
                valid_length3, segment_ids3):
        attention_mask1 = self.gen_attention_mask(token_ids, valid_length)
        attention_mask2 = self.gen_attention_mask(token_ids2, valid_length2)
        attention_mask3 = self.gen_attention_mask(token_ids3, valid_length3)

        _, pooler = self.bert1(input_ids=token_ids, token_type_ids=segment_ids.long(),
                               attention_mask=attention_mask1.float().to(token_ids.device))
        _, pooler2 = self.bert2(input_ids=token_ids2, token_type_ids=segment_ids2.long(),
                                attention_mask=attention_mask2.float().to(token_ids2.device))
        _, pooler3 = self.bert3(input_ids=token_ids3, token_type_ids=segment_ids3.long(),
                                attention_mask=attention_mask3.float().to(token_ids3.device))
        if self.dr_rate:
            out = self.dropout(pooler)
            out2 = self.dropout(pooler2)
            out3 = self.dropout(pooler3)
        out = self.classifier1(out)
        out2 = self.classifier2(out2)
        out3 = self.classifier3(out3)
        cat = torch.cat((out, out2, out3), dim=1)
        fc_out = F.relu(self.fc1(cat))
        output = self.fc2(fc_out)

        return output
```

### How to use
<pre>
<code>
python main.py --maxlen1 [과제명 seqence length] --maxlen2 [연구목표 sequence length] --maxlen3 [키워드 sequence length] --batch_size [int] --epochs [int] --loss [CE of FC]
</code>
</pre>


## 2. ENN BERT

- 하나의 BERT모델을 활용해서 유의미 하다고 판단한 3가지 column을 합쳐서 입력했다.
- class의 imbalance가 매우 심한 데이터 였다.
- 모델에 학습시키기전 undersampling 작업을 거치기 위해 ENN을 적용했다.

```python
def ENN(token_vectors, dataset_train):
    sampler = EditedNearestNeighbours()
    X_res, y_res = sampler.fit_resample(token_vectors, dataset_train['binary'])
    return X_res, y_res
```

### How to use
<pre>
<code>
python main.py --maxlen1 256 --batch_size [int] --epochs [int] --loss [CE of FC] --sampling [ENN or SMOTEENN]
</code>
</pre>

## Focal Loss

- 본 과제에서는 Data Imbalance 문제가 있을 경우에 많이 사용되는 "Focal Loss"를 사용했다.
- Focal loss 에서는 마지막에 출력되는 각 클래스의 probability를 이용해 CE Loss에 통과된 최종 확률값이 큰 EASY 케이스의 Loss를 크게 줄이고 최종 확률 값이 낮은 HARD 케이스의 Loss를 낮게 줄이는 역할을 한다. 
- 보통 CE는 확률이 낮은 케이스에 패널티를 주는 역할만 하고 확률이 높은 케이스에 어떠한 보상도 주지만 Focal Loss는 확률이 높은 케이스에는 확률이 낮은 케이스 보다 Loss를 더 크게 낮추는 보상을 주는 차이점이 있다.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        nn.CrossEntropyLoss()
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
```

### 결과
- 결과는 정확도가 최대 약 76% 정도로 좋지 않은 성능을 보였다.
- BERT를 이용한 분류모델을 이용해서 여러가지 실험을 해볼 수 있는 기회였다.
