import torch
from torch import nn
import torch.nn.functional as F



class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 num_classes,
                 hidden_size=768,
                 h1=128,
                 h2=256,
                 h3=128,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert1 = bert
        self.bert2 = bert
        self.bert3 = bert
        self.dr_rate = dr_rate

        self.classifier1 = nn.Linear(hidden_size, h1)
        self.classifier2 = nn.Linear(hidden_size, h2)
        self.classifier3 = nn.Linear(hidden_size, h3)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
        self.fc1 = nn.Linear(h1 + h2 + h3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

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

if __name__ == '__main__':
    from kobert.pytorch_kobert import get_pytorch_kobert_model

    bertmodel, vocab = get_pytorch_kobert_model()
    model = BERTClassifier(bertmodel, 46, dr_rate=0.5)
    print(model)