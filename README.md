![](https://upload-images.jianshu.io/upload_images/13575947-0cf594843461afd4.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 机器人写手的时代来临

你可能听说过，2015年腾讯推出了一个叫Dreamwriter的自动化新闻写作机器人，它的成文速度仅为0.5秒，一眨眼功夫就能产出十几篇文章，这是相当惊人的。

现阶段的写作机器人大多应用于体育和财经新闻。Why? 一方面是因为这些领域包含大量数据起伏，比如股价波动、比赛成绩等。另一方面，对于这类报道，读者更关心的是具体数字而非语言本身，他们对古板的新闻套话已有较高的容忍度。

我们要开发的写作机器人，是通过语言模型（Language Model）来学习英文影评写作的。以下两段对话分别是我和机器人写的影评，可以清楚地看到，机器人（Bot）写的东西虽然笨拙，但好在主题明确，没有语法硬伤，看起来写的有模有样，有点儿小学生作文的味道😄。

> **\>**  It's not a movie, it's a video comedy. A lot of trash and a lot of humor.
**Bot:**  i'm sure you will be disappointed. the only thing that i didn't like was the fact that the movie wasn't funny.

>**\>**  God is really the reversal of the goose bumps
**Bot:**  i think that the movie is a bit too long, but it is a very good movie. i have to say that i was surprised to see that this movie was made in the same vein as "the blair witch project". 

完整的代码见： [Notebook](https://github.com/alexshuang/writingbot/blob/master/writingbot.ipynb) /  [Github](https://github.com/alexshuang/writingbot)。

---

## [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

[IMDB](https://baike.baidu.com/item/%E4%BA%92%E8%81%94%E7%BD%91%E7%94%B5%E5%BD%B1%E8%B5%84%E6%96%99%E5%BA%93/1484580?fromtitle=IMDB&fromid=925061&fr=aladdin)是类似国内的豆瓣电影的网站，里面有大量用户的观影评论。我们要用的是斯坦福大学的[NLP情绪分析数据集](http://ai.stanford.edu/~amaas/data/sentiment/)，它收录有100k条IMDB影评。

```
trn_txts = read_text(TRN_PATH)
val_txts = read_text(VAL_PATH)
txts = np.concatenate([trn_txts, val_txts])
df = pd.DataFrame({'text': txts})
```

数据集中的影评是逐条以文本文件的格式存储的，因此，需要先把它们全部读取汇总起来，生成panda's dataframe，便于后续对数据进行预处理。

### Preprocess: Tokenize & Numericalize

我们知道，机器学习模型的输入只能是数值（int/long/float），因此，文本预处理中最主要的工作就是将文字（str）转换为数值。这个过程就称为数值化（numericalize）。

数值化的前提是句子（sentence）已经被拆分成片段（token）。例如，“good morning!”这个句子，假设以单词为单位拆分，将得到 ["good", " ", "morning", "!"] token序列。这个过程就称为tokenize。

首先，我们先对数据集的所有文本做tokenize，这需要用到Tokenizer类。

```
class Tokenizer():
  def __init__(self, lang='en'):
    self.tok = spacy.load(lang)

    ......
  @staticmethod
  def tokenize_mp(x, n_cpus=None, lang='en'):
    ......
    with ThreadPoolExecutor(n_cpus) as e:
      toks += sum(e.map(Tokenizer(lang).tokenize, xs), [])
    return toks

  @staticmethod
  def tokenize_df(dl, label, lang='en'):
    ts = []
    for i, df in enumerate(dl):
      ts += Tokenizer.tokenize_mp(df.iloc[:, label].values, lang=lang)
    return ts
```
Tokenizer，它封装了tokenize工具--spacy，并支持多线程处理（tokenize_mp）。NLP数据集的特点是数据量庞大（本例的单词数超过2400万），因此，它们的预处理就对**CPU算力**和**内存大小**提出了更高的要求。

```
chunksize = 25000
dl = pd.read_csv(PATH/'txts.csv', header=None, chunksize=chunksize)
ts = Tokenizer.tokenize_df(dl, 0)
```

除了CPU算力，**内存不足**则是另一个让NLPer头痛不已的难题。我们这里采用分步加载数据集的方式来避免内存不足，pd.read_csv()不直接读取数据，而是生成分块（chunksize）读取数据的generator迭代器。

```
freq = Counter(p for o in trn_toks for p in o)
freq.most_common(5)

[('the', 1208826),
 (',', 986530),
 ('.', 985698),
 ('and', 587158),
 ('a', 583198)]

vocab_size = 60000
min_freq = 4
itos = [n for n, v in freq.most_common(vocab_size) if v >= min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')
stoi = collections.defaultdict(lambda: 0, {o:i for i, o in enumerate(itos)})
```

tokenize完成之后，就要对数据集的文本内容做numericalize。numericalize的过程就有点像谍战剧中情报员解码密文的桥段，情报员收到一串数字或乱码，连忙掏出密码本，逐个查找密文对应的明文。

numbericalize对token序列编码的过程也需要这样一本“密码本” -- token字典（stoi），它记录了所有出现4次以上的高频token以及它们对应的数值，字典最大长度是60000，那些不存在于字典中的token会被标记为unknown（\_unk\_）。

### Dataset

除了预处理之外，文本序列还需要转换为RNN需要的结构：

![Figure 1](https://upload-images.jianshu.io/upload_images/13575947-bf55caaa1ea7c1a0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如Figure1所示，词序列被划分为**batch_size**个段，这些段经过叠加翻转后得到一个矩阵，而mini-batch就是矩阵中**bptt**个连续行组成的子矩阵。

RNN input（mini-batch） shape是**[seq_len, batch, input_size]**：**seq_len**是mini-batch的行数--BPTT（Back Propagation Through Time），**batch**是batch_size，**input_size**是词向量长度。

Figure1中的mini-batch并没有“**input_size**”这个维度，那么，词向量是从何而来的？神经网络不是万能的，它很难通过观察一个数字就能学会数字背后代表的单词、单词间的关联以及语法结构。

例如“ friday”、"saturday"和"sunday"这三个单词，它们的编码分别是5、6和100，你很难想象如何通过观察这些数字来学习这些词的意思，不能因为5和6相邻就判定星期五和星期六的关联度最强。实际上，每个单词都有很多明特征和暗特征，例如"saturday"的明特征是“周末”，暗特征是“购物”、“娱乐”、“社交”等。

既然一个数不够，那就用一个向量的数来表征一个词吧。这个向量就称为词向量，两个向量越相近，就表示两个词越相关，反之亦然。

与此同时，token字典的用途也变成了从词向量矩阵中查找词向量。因为词向量是嵌入（embedding）到mini-batch的（想想查字典），因此词向量词典也称为词嵌入矩阵（word embedding matrix）。

```
class ModelDataloader():
  def __init__(self, nums, bs, bptt):
    self.bs, self.bptt = bs, bptt
    self.data = self.batchify(nums)
    self.n = self.data.size(0)

  def __next__(self):
    ......
    
  def batchify(self, data):
    ......

class ModelData():
  def __init__(self, vocab, bs, bptt, trn_ds, val_ds, test_ds=None):
    self.trn_ds, self.val_ds, self.test_ds = trn_ds, val_ds, test_ds
    self.vocab, self.n = vocab, len(vocab)
    self.trn_dl = ModelDataloader(self.trn_ds, bs, bptt)
    self.val_dl = ModelDataloader(self.val_ds, bs, bptt)
    self.test_dl = None if self.test_ds is None else ModelDataloader(self.test_ds, bs, bptt)
```

ModelDataloader.batchify()用于将文本序列转成矩阵，ModelDataloader.\_\_next__()用于读取mini-batch。

---

## RNN Language Model

![Figure 3: N-gram, language model](https://upload-images.jianshu.io/upload_images/13575947-2a7d4ad12209cdab.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如Figure3所示，语言模型就是通过前面N-1个词来预测第N个词（N-gram）。

![Figure 4: https://arxiv.org/abs/1611.01462](https://upload-images.jianshu.io/upload_images/13575947-525ff7ed8531b65e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Figure4中三条公式简洁明了地描述了RNN语言模型的神经网络架构：
- 将embedding矩阵中的词向量嵌入mini-batch，得到$x$。
- 将输入$x$和上一个$time$的hidden state（$h_{t-1}$）一起传入RNN，得到当前$time$的hidden state（$h_{t}$）。
- 将$h_{t}$传入分类器（linear layer），并通过softmax函数生成输出，概率最大的分类就是当前$time$的预测词。

初始模型--RNNLM（RNN language model），就是上述这段描述的是代码实现：
```
vocab_size = md.n
emb_nf = 200
nf = 256

class RNNLM(nn.Module):
  def __init__(self, vocab_size, emb_nf, nf):
    super().__init__()
    self.vocab_size = vocab_size
    self.emb = nn.Embedding(vocab_size, emb_nf)
    self.i_h = nn.Linear(emb_nf, nf)
    self.rnn = nn.RNN(nf, nf)
    self.h_o = nn.Linear(nf, vocab_size)
    self.h = torch.zeros(1, bs, nf).cuda()
    self.tanh = nn.Tanh()
    
  def forward(self, x):
    x = self.tanh(self.i_h(self.emb(x)))
    o, h = self.rnn(x, self.h)
    self.h = h.detach()
    return F.log_softmax(self.h_o(o), -1).view(-1, self.vocab_size)
```
代码**“self.h = h.detach()”**的意思是，下一个mini-batch的训练可以以在当前mini-batch的学习成果为基础。

### Train

```
def validate(stepper, dl, metrics):
      ......
    t = tqdm(iter(dl), leave=False, total=len(dl), miniters=0)
    for xs, y in t:
      preds, loss = stepper.validate(xs, y)
      ......

class Stepper():
  ......
def step(self, xs, y):
    preds = self.m(xs)
    loss = self.crit(preds, y)
    self.opt.zero_grad()
    loss.backward()
    if self.clip: nn.utils.clip_grad_norm_(get_trainable_parameters(m), self.clip)
    self.opt.step()
    return loss.data.item()
  ......

def fit(model, data, epochs, opt, crit, clip=0., metrics=None):
  ......
  for ep in tnrange(epochs, desc='Epochs'):
    ......
    # train
    t = tqdm(iter(data.trn_dl), leave=False, total=len(data.trn_dl), miniters=0)
    for xs, y in t:
      loss = stepper.step(xs, y)
      t.set_postfix(loss=loss, refresh=False)
    t.close()

    # validation
    values = validate(stepper, data.val_dl, metrics)
    ......
```

Train模块--fit()的架构和设计，很大一部分我是参考了[Fastai library](https://github.com/fastai/fastai/blob/master/old/fastai/model.py)，当然，我也无耻地从中copy了部分代码😎。如果你不熟悉pytorch，建议先到pytorch官网学习相关教程。

```
fit(m, md, 1, opt, F.nll_loss)

100% 1/1 [19:59<00:00, 1199.91s/it]
     epoch      trn_loss   val_loss  
       0        4.882467   4.941533  
```
到此，RNNLM的首秀结束，一切都运行良好。但实际上，fit()存在一个问题：没有学习率退火（learning rate annealing），即在整个训练过程中，学习率固定不变。

我们知道，学习率决定小球在优化曲线上的运动步距，要想让小球落在曲线最低点，那么应该随着训练过程的深入，逐步减小学习率，缩小小球运动步距，像Figure3中的上图。反之，如果学习率不减反增，那小球的运动轨迹就会像Figure3中下图那样，在低点无法收敛，反而向高点反弹。

![Figure 3](https://upload-images.jianshu.io/upload_images/13575947-cebd833fd9b1c20c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

学习率固定不变，可以预见到，模型不可能会收敛到最低点，会在曲线低点附近来回震荡。因此，我采用[SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)来训练模型，因为篇幅原因我决定把它的介绍放到另一篇博客，坑名我都想好了，[你应该知道的神经网络训练大法（待填）]()。

如Figure4所示，SGDR会在每个mini-batch训练结束后，按照cosine曲线来减小学习率，并在每个epoch开始训练前restart学习率到最大值，并将待训练的mini-batch数翻倍。
![Figure 4](https://upload-images.jianshu.io/upload_images/13575947-a87342130466fda9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
class CosAnneal(LRUpdater):
  ......
  def on_cycle_begin(self):
    super().on_cycle_begin()
    self.t_cur = 0
    
  def calc_lr(self):
    new_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(self.t_cur / self.nb * np.pi))
    self.t_cur += 1
    if self.t_cur == self.nb:
      self.t_cur = 0
      self.nb *= self.cycle_mult
    return new_lr

def fit(model, data, epochs, opt, crit, clip=0, metrics=None, callbacks=None):
  ......
      for xs, y in t:
        for cb in callbacks: cb.on_batch_begin()
        loss = stepper.step(xs, y)
        for cb in callbacks: cb.on_batch_end(loss)
    ......
```

CosAnneal就是SGDR的学习率退火函数，它会在每个mini-batch训练结束后（cb.on_batch_end()）调整学习率。

```
fit(m, md, epochs, opt, F.nll_loss, callbacks=[cosanneal])

100% 2/2 [1:00:04<00:00, 1560.49s/it] 
epoch      trn_loss   val_loss  
       0        4.965671   5.012351       
       1        4.839868   4.848449  
       1        4.779568   4.7841 
```

可以看到，模型已经可以收敛了，但存在过拟合倾向。每个epoch的训练结果，都是train loss小于validation loss，那至少说明模型容易过拟合。

到此，RNNLM该退场了，接下来时间交给我们另一个主角--weight-dropped LSTM。

## weight-dropped LSTM

[weight-dropped LSTM](https://arxiv.org/abs/1708.02182)的作者Stephen Merity提出了一个洞见：将CNN的标配，专用于对抗过拟合的dropout应用于RNN模型。这就是weight-dropped LSTM。

weight-dropped LSTM和RNNLM的区别在于，除了隐层是由3个LSTM层组成之外，模型中的各层都增加了dropout。这段代码太长，我就不贴了，[请点这里查看](https://github.com/alexshuang/writingbot/blob/master/writingbot.ipynb)。

模型中的dropout有三种：
- **embbeding dropout**：顾名思义，它是作用于embedding矩阵的dropout，它除了dropout功能之外，还会将non-dropped-out向量scale为它的$1/(1-p)$倍，p是dropout的概率。
- **variational dropout**：它作用于层与层之间，作用于hidden state，换句话说，它抛弃掉了一个mini-batch中某些词的学习结果（hidden state）。跟embbeding dropout一样，都是从token的层面来进行dropout。
- **weight dropout**：它作用于LSTM层，作用于LSTM层内部的神经元权值。这里不是直接修改pytorch LSTM类的代码，而是通过创建LSTM的wrapper类--WeightDrop，由它通过委托（delegate）来调用LSTM.forward()，如此就可以曲线对LSTM内部的权值做dropout。

除了dropout，这篇[paper](https://arxiv.org/abs/1708.02182)在模型训练过程中还用到一个很创新的点：BPTT（Back Propagation Through Time）。眼熟吧，它在前面**Dataset**模块中提到，不记得的可以回Figure1看一眼。

文本数据的两个特性：顺序性和固定的语法结构，使得RNN既不能像CNN那样，可以打乱训练样本的排列顺序（shuffle），也不能轻易地做数据扩充（data augmentation）。这样一来，喂给RNN的数据就缺乏了变化和随机，使得模型更容易过拟合。

为此，weight-dropped LSTM在读取mini-batch时，seq_len不是固定的，它有95%的概率落在$\mu=bptt（超参）, \sigma=5$的分布中，假设bptt=70，那么seq_len有90%的概率落在[55，85]。另外还有5%的概率，bptt会减半，即bptt/2。

```
emb_sz = 200
nf = 500
nl = 3
clip = 0.25
ps = np.array([0.05, 0.05, 0.05, 0.02, 0.1]) * 1
lr = 3e-3
cycle_len = 1
cycle_mult = 2
epochs = 3
epochs = [cycle_len * cycle_mult**i for i, _ in enumerate(range(epochs))]

m = WDLSTM(bs, vocab_sz, emb_sz, pad_idx, nf, nl, ps).cuda()
opt = optim.Adam(m.parameters(), lr=lr, betas=(0.75, 0.99))
cosanneal = CosAnneal(opt, len(md.trn_dl), cycle_len=cycle_len, cycle_mult=cycle_mult)
fit(m, md, epochs, opt, F.cross_entropy, clip=clip, callbacks=[cosanneal], metrics=[accuracy])

HBox(children=(IntProgress(value=0, description='Epochs', max=3, style=ProgressStyle(description_width='initia…
     epoch      trn_loss   val_loss   accuracy  
       0        4.882206   4.794618   0.239594  
       1        4.812892   4.651131   0.247563  
       1        4.529045   4.584595   0.254397  
       2        4.523168   4.593299   0.250682  
       2        4.562455   4.524473   0.256603  
       2        4.374611   4.481817   0.261064  
       2        4.486718   4.47125    0.262309 
```
可以清楚地看到，模型的收敛速度更快了，过拟合也消失了。更惊喜的是，虽然模型的复杂度成倍增加了，但训练时间却并没有因此增加。模型完美地展现了深度神经网络对抗过拟合的正确姿势：**增加正则化（dropout和weight decay），而非降低神经网络复杂度**。

## Testing

到这里，我们已经完成了模型的初步训练，接下来开始进行模型测试。我在豆瓣电影上随意选了几句影评，经有道词典翻译后丢给模型，让它循环生成最长为500词的内容。为简化演示，默认只输出生成的第一个句子，如果需要输出更多的文本，可以通过参数**sentences**指定。

```
s = 'God is really the reversal of the goose bumps'
res = keep_writing(m, s, sentences=2)
print("> ", s)
print("Bot: ", res)

>  God is really the reversal of the goose bumps
Bot:  i think that the movie is a bit too long, but it is a very good movie. i have to say that i was surprised to see that this movie was made in the same vein as "the blair witch project". 
```

可以看到，正如文章开头所描述的那样，机器人生成的内容没有语法硬伤，标点符号也用的精确（尤其是双引号），虽然没有什么文采，但好歹主题没跑偏。

总的来说，对于这个结果我还是基本满意的，接下要做的就是继续提高模型的预测准确率。除了继续训练模型直到过拟合之外，还可以通过预训练和迁移训练来给模型赋能。

## Pretrain: fasttext

我相信你一定听说过鼎鼎大名的word2voc，实际上，它就是embedding矩阵，只是它经过在wikipedia等大型语料库的训练后，词向量不再是随机值。换句话说，word2voc的预训练已经帮模型学会了英文单词，在此基础上再学习英文的语法，效果自然是事半功倍。

相比古老的word2voc，fasttext是更好的选择，它经过更多更大的语料库训练，支持的语言也更多。

不知道你发现没有，不管是word2voc还是fasttext，都有个弊端：它们只是训练了embedding矩阵，而它只占模型的一小部分。与这种训练相反的是迁移训练，即先用相同的语料库来训练模型，再用IMDB语料库来fine-tune模型，这样的效果最好，但整个训练就需要花费很长很长的时间。

正所谓进一寸有一寸的欢喜，这里我们先用fasttext预训练来感受下进步的喜悦。

[Fasttext](https://fasttext.cc/)提供了两种数据格式：bin、txt，前者的加载速度比后者快得多，如果你机器内存足够，建议用前者。如果你有内存不足的困扰，可以从txt文件读取，但用时会比前者要长得多。

```
m = WDLSTM(bs, vocab_sz, emb_sz, pad_idx, nf, nl, ps, emb_weights=emb_w).cuda()
fit(m, md, epochs, opt, F.cross_entropy, clip=clip, callbacks=[cosanneal], metrics=[accuracy])

HBox(children=(IntProgress(value=0, description='Epochs', max=3, style=ProgressStyle(description_width='initia…
     epoch      trn_loss   val_loss   accuracy  
       0        4.579641   4.529013   0.252729  
       1        4.548646   4.438005   0.260425  
       1        4.379018   4.365056   0.267696  
       2        4.414107   4.4228     0.261393  
       2        4.526043   4.361377   0.267618  
       2        4.145617   4.318527   0.272548  
       2        4.295489   4.305556   0.273613  
```

可以清楚地看到，使用预训练的词向量后，经过相同的训练，模型的准确率提高了1%，效果不错。

### END

本文详细介绍了如何用语言模型从零打造IMDB影评机器人。语言模型给我的感觉是，训练成本很低，不需要对数据做标注，随便换个数据集就可以变身为AIxx写作机器人。我下一篇博客要用seq2seq模型开发聊天机器人，届时再来对比两个模型的优劣，希望能进一步优化该模型。
