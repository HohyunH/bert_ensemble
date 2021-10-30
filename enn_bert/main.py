import torch
from torch import nn
from DataLoader import BERTDataset
import os
from tqdm import tqdm
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import pickle
import argparse

import Sampling
import BERT


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


def evaluate(model, eval_loader, device):
    model.eval()
    train_acc = 0.0
    with torch.no_grad():
        for batch_id, ((token_ids, valid_length, segment_ids, label), binary) in enumerate(tqdm(eval_loader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            label = label.long().to(device)
            valid_length = valid_length
            out = model(token_ids, valid_length, segment_ids)
            train_acc += calc_accuracy(out, label)
    print("ACC : ", train_acc / (batch_id + 1))


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


def main(args):
    # Setting parameters
    max_len = args.max_len
    batch_size = args.batch_size
    warmup_ratio = args.warmup_ratio
    num_epochs = args.epochs
    max_grad_norm = args.grad_norm
    log_interval = args.log_interval
    learning_rate = args.learning_rate

    device = torch.device(f"{args.gpu}:0")

    # load resampled data
    # dataset_train, dataset_test, sample_submission = Sampling.preprocessing(
    #     '/home/oldman/climate/data')
    dataset_train, dataset_test, sample_submission = Sampling.preprocessing(
        '/Users/oldman/PycharmProjects/climate/data')
    bert_model, tok, transform = Sampling.transforming(max_len=max_len)
    token_vectors = Sampling.tokenizing(transform, dataset_train)
    if os.path.isfile(f'./{args.sampling}_X_{max_len}.pickle'):
        with open(f'./{args.sampling}_X_{max_len}.pickle', 'rb') as file:
            X_res = pickle.load(file)
        with open(f'./{args.sampling}_y_{max_len}.pickle', 'rb') as file:
            y_res = pickle.load(file)
    else:
        if args.sampling == 'ENN':
            X_res, y_res = Sampling.ENN(token_vectors, dataset_train)
        elif args.sampling == 'SMOTEENN':
            X_res, y_res = Sampling.SMOTEENN(token_vectors, dataset_train)

        with open(f'./{args.sampling}_X_{max_len}.pickle', 'wb') as file:
            pickle.dump(X_res, file)
        with open(f'./{args.sampling}_y_{max_len}.pickle', 'wb') as file:
            pickle.dump(y_res, file)

    # find idx / preprocessing
    train_idx = []
    j = 0
    for i, x in enumerate(tqdm(token_vectors)):
        if (x == X_res[j]).sum() == max_len:
            train_idx.append(i)
            j += 1
    one_st = dataset_train.index[dataset_train['binary'] == 1].tolist()
    train_idx.extend(one_st)
    train_idx.sort()
    train_data = dataset_train.loc[train_idx].reset_index()
    train_dataset, val_dataset = train_data[:100000], train_data[100000:]

    # data loader
    data_train = BERTDataset(dataset=train_dataset, bert_tokenizer=tok, max_len=max_len, pad=True, pair=False, test=False)
    data_val = BERTDataset(dataset=val_dataset, bert_tokenizer=tok, max_len=max_len, pad=True, pair=False, test=False)
    data_test = BERTDataset(dataset=dataset_test, bert_tokenizer=tok, max_len=max_len, pad=True, pair=False, test=True)
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5, shuffle=False)

    # model
    model = BERT.BERTClassifier(bert_model, num_classes=46, dr_rate=0.1).to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = FocalLoss()
    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    # train
    for e in range(num_epochs):
        train_acc = 0.0
        model.train()
        for batch_id, ((token_ids, valid_length, segment_ids, label), _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            # binary = binary.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print(
                    f"epoch {e + 1} batch id {batch_id + 1} loss {loss.data.cpu().numpy()} train acc {train_acc / (batch_id + 1)}")
                # torch.save(model.state_dict(), f"/home/oldman/climate/bert_model_{args.max_len}.pt")
                torch.save(model.state_dict(), f"/Users/oldman/PycharmProjects/climate/BERT_{args.sampleing}_{args.max_len}.pt")
        print(f"epoch {e + 1} train acc {train_acc / (batch_id + 1)}")
        evaluate(model, val_dataloader, device)

    # test
    answers = []
    model.load_state_dict(torch.load(f'/Users/oldman/PycharmProjects/climate/BERT_{args.sampleing}_{args.max_len}.pt'))
    model.eval()
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            out = model(token_ids, valid_length, segment_ids)
            answers.extend(torch.max(out, 1)[1].tolist())

    # results
    sample_submission['label'] = answers
    sample_submission.to_csv('/Users/oldman/PycharmProjects/climate/bert_baseline_256.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    parser.add_argument('--max-len', '-ml', type=int, default=256, help='Max length')
    parser.add_argument('--batch-size', '-bs', type=int, default=24, help='Batch size')
    parser.add_argument('--warmup-ratio', '-wr',type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--epochs', '-ep', type=int, default=10, help='Number of epochs')
    parser.add_argument('--grad-norm', '-gn', type=int, default=1, help='Max gradient norm')
    parser.add_argument('--log-interval', '-li', type=int, default=200, help='Log interval')
    parser.add_argument('--learning-rate', '-lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--gpu', '-gpu', type=str, default='cuda', help='GPU(cuda) OR CPU(cpu)')
    parser.add_argument('--sampling', '-sp', type=str, default='ENN', help='ENN or SMOTEENN')
    args = parser.parse_args()

    main(args)