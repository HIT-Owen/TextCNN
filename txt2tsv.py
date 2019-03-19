import jieba
label2id = {'体育': '0', '娱乐': '1', '家居': '2', '房产': '3', '教育': '4',
            '时尚': '5', '时政': '6', '游戏': '7', '科技': '8', '财经': '9'}

ft = open('data/train.tsv', 'w')
fv = open('data/valid.tsv', 'w')
ft.write('label\ttext\n')
fv.write('label\ttext\n')

with open('data/cnews.train.txt') as f:
    num = 0
    for line in f:
        num += 1
        print(num)
        line = line.split('\t')
        line[0] = label2id.get(line[0])
        line[1] = ' '.join(jieba.cut(line[1]))
        line = '\t'.join(line)
        if num % 10 == 0:
            fv.write(line)
        else:
            ft.write(line)


ft.close()
fv.close()
