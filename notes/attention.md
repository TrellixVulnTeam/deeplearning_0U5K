# 深度学习中有益的概念

在面向不同的问题时，你所学习的知识的重要程度是有所差异的，比如曾经花费很多心血弄懂的卷积网络，在目标检测算法中仅仅是作为预训练特征提取而被一言带过，而有些你认为不重要的内容，却不管是在如何复杂的模型中，总是作为不可或缺的成分出现，比如深度学习中分别面向分类和回归的模型，不管表示部分的结构如何复杂，总而言之是为最终的分类回归问题服务的，而那几个经典的回归，则适时的出现，补上解决问题的最后一块拼图。

## Logistic Regression

### Binomial Logistic Regression 

一个事件的几率(odds)是指该事件发生的概率与该事件不发生的概率的比值。如果事件发生的概率是p，那么事件的几率是$\frac{p}{1-p}$,该事件的对数几率(log odds)或logit函数是$logit(p) = log\frac{p}{1-p}$.对Logistic Regression而言：
$$
log\frac{p(Y=1|X)}{1-p(Y=1|X)}=\vec{\omega}\cdot \vec{x}
$$
输出Y=1的对数几率是输入x的线性模型，使用极大似然估计法估计参数模型，求的似然函数最大时，$\vec\omega$的值.
hi

## CNN

**输入及中间特征图**

卷积层堆叠特征图与定义卷积层的张量有关，一个常规卷积网络的输入（可以由一个3-D tensor定义，[height,width, channels],则一个mini-batch 则可由一个4-D张量定义，4-D tensor of shape [mini-batchsize, height, width, channels]）。



**卷积层参数**

卷积层的参数也是用一个4D的张量表示

4-D tensor of shape [fh, fw, fn, fn′]，偏移项是 1-D tensor of shape [fn]张量的四个维度分别代表：前2核大小，前一层特征图数或输入通道数（同时逐一对应一个偏移项），最末是输出特征图数目

 

**具体每个特征图的大小则由多个因素共同决定**.



## 评价

|      | T    | F    |
| ---- | ---- | ---- |
| T    | TP   | FN   |
| F    | FP   | TN   |

精确率定义为：P=TP/（TP+FP）

召回率定义为：R=TP/（TP+FN）