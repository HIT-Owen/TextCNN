# TextCNN
TextCNN by using Pytorch1.0
python3.6 + torchtext0.4.0

语料采用清华的新闻语料（精简后10个类别，每个类别5000样本）  
语料已经采用jieba分词并保存成tsv格式(你也可以采用其他工具分词，在torchtext里面分词速度较慢）  
without any pre-trained word embedding  
  
![笔记本CPU用验证集训练的结果](https://github.com/HIT-Owen/TextCNN/blob/master/images/2019-03-19%2016-26-56%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

[torchtext-tutorial](http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/)  
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
