import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate') 
parser.add_argument('--epochs', type=int, default=10, help='training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--emb_size', type=int, default=64, help='embedding size') 
parser.add_argument('--class_num', type=int, default=10, help='class num')
parser.add_argument('--kernel_num', type=int, default=5, help='conv kernel num')
parser.add_argument('--kernel_sizes', type=list, default=[3,5,7], help='kernel size')
parser.add_argument('--log_interval', type=int, default=50, help='print steps')
parser.add_argument('--eval_interval', type=int, default=300, help='eval steps')
parser.add_argument('--save_interval', type=int, default=500, help='save steps')
parser.add_argument('--save_dir', default='model/', help='save dir')
parser.add_argument('--save_best', type=bool, default=True, help='save with best acc')
args = parser.parse_args()

# print(type(args.kernel_sizes))

label2id = {'体育': '0', '娱乐': '1', '家居': '2', '房产': '3', '教育': '4',
            '时尚': '5', '时政': '6', '游戏': '7', '科技': '8', '财经': '9'}

# 定义字段处理方式
tokenize = lambda x: x.split()

print('building stop words...')
stop_words = []
with open('data/stopwords.txt') as f:
    for l in f:
        stop_words.append(l.strip())

TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, stop_words=stop_words, batch_first=True)
LABEL = data.Field(sequential=False, use_vocab=False, preprocessing=lambda x: int(x))

print('building dataset...')
# 生成数据集
trn, vld = data.TabularDataset.splits(
    path='data',
    train='train.tsv', validation='valid.tsv',
    format='tsv',
    skip_header=True,
    fields=[('label', LABEL), ('text', TEXT)])

# exm = trn[0]
# print(exm.text)
# print(exm.label)
# print(type(exm.label))

# 建立词汇表
print('building vocab...')
TEXT.build_vocab(vld)
# print(len(TEXT.vocab))
# print(TEXT.vocab.freqs.most_common(5))

# 生成迭代器
trn_iter, vld_iter = data.BucketIterator.splits(
    (trn, vld),
    batch_sizes=(args.batch_size, args.batch_size),
    device=torch.device('cpu'),
    sort_key=lambda x: len(x.text),
    sort_within_batch=False,
    repeat=False
)

# batch = next(iter(trn_iter))
# print(batch.label)
# print(batch.text.shape)
# print(batch.__dict__.keys())


class TextCNN(nn.Module):
    def __init__(self, vocab_size, args):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = args.emb_size
        self.class_num = args.class_num
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes

        self.embed = nn.Embedding(self.vocab_size, self.emb_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (ksize, self.emb_size)) for ksize in self.kernel_sizes])
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(len(self.kernel_sizes)*self.kernel_num, self.class_num)

    def forward(self, x):
        x = self.embed(x) # batch_size seq_len emb_size
        x = x.unsqueeze(1) # batch_size 1 seq_len emb_size

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        # batch_size, kernel_num, seq_len-kernel_size+1, (1)
        x = [F.max_pool1d(line, line.shape[2]).squeeze(2) for line in x]
        # batch_size, kernel_num, (1)

        x = torch.cat(x, 1) # batch_size, kernel_num*len(kernel_sizes)
        x = self.drop(x) 

        logit = self.fc(x) # batch_size, class_num
        return logit


def train(optimizer, loss_func, train_iter, valid_iter, model, args):
    # model = TextCNN()
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # loss_func = nn.CrossEntropyLoss()

    steps = 0
    best_acc = 0
    last_step = 0

    model.train()
    print('training...')

    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label 

            logit = model(feature)
            loss = loss_func(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                result = torch.max(logit, 1)[1].view(target.size())
                corrects = (result.data == target.data).sum()
                accuracy = corrects*100.0/batch.batch_size
                print(f"Epoch[{epoch}] - Batch[{steps}] - loss: {loss.item():.3f} - acc: {accuracy: .0f}%")

            if steps % args.eval_interval == 0:
                eval_acc = eval(loss_func, valid_iter, model)
                if eval_acc > best_acc:
                    best_acc = eval_acc
                    last_step = steps
                    if args.save_best:
                        save_model(model,args.save_dir,'best',steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print(f"early stop by {args.early_stop} steps.")

            elif steps % args.save_interval == 0: ##保存模型
                save_model(model,args.save_dir,'snapshot',steps)

def eval(loss_func, valid_iter, model):
    model.eval()

    corrects, avg_loss = 0, 0
    for batch in valid_iter:
        feature, target = batch.text, batch.label 
        logit = model(feature)
        loss = loss_func(logit, target)

        avg_loss += loss.item()
        result = torch.max(logit, 1)[1].view(target.size())
        corrects += (result.data == target.data).sum()

    total_size = len(valid_iter.dataset)
    avg_loss /= total_size
    accuracy = corrects*100.0/total_size
    print(f"\nEvaluation - loss: {avg_loss:.3f} - acc: {accuracy: .0f}%")
    return accuracy


def save_model(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = f"{save_prefix}_steps_{steps}.pt"
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    model = TextCNN(len(TEXT.vocab), args)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    loss_func = nn.CrossEntropyLoss()
    train(optimizer, loss_func, vld_iter, vld_iter, model, args)



