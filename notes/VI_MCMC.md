## 从VAE谈变分推理和蒙特卡罗采样

> 这不是一篇科普文章，用几个形象的比喻解释概念就结束了，做好准备去真正理解几个公式，不同方法间的联系和区别也就显而易见了。

像往常一样，我们从贝叶斯公式开始：
$$
p(z|x) = \cfrac{p(x|z)p(z)}{p(x)}\tag{1}
$$
引用一个很好的解释，如果通过$x \in R^D$上的概率分布对整个世界建模，其中$p(x)$表示$x$可能处于的状态。这个世界可能非常的复杂，我们无法知道$p(x)$的具体形式。为了解决这个问题，引入另一个变量$z\in R^d$来描述$x$的背景信息，这个变量使得我们可以将$p(x)$表示为一个无限混合模型：
$$
p(x) = \int p(x|z)p(z)dz\tag{2}
$$
对于$z$的任意可能，都引入另一个条件分布，并通过$z$的概率进行加权，最终得到$p(x)$取值。那么基于上面这样的设定，那么我们需要解决的问题就变成：给定$x$的观测值，隐变量$z$是什么，即贝叶斯公式的左边——后验概率分布$p(z|x)$。



无论是计算后验概率分布$p(z|x)$还是边缘概率分布$p(x)$都需要计算式$(2)$，但是我们需要知道，由于$x$和$z$的取值空间都非常大，通常情况我们认为是无法直接计算的，因而需要采用近似推理进行估计。



事实上，我们有几种不同的方法可以对$p(x)$进行估计，这里我们重点介绍和对比两种方式，变分推理(VI，varitational inference)和MCMC(Marokv Chain Monte Carlo)。

根据重要采样，当我们需要对原始（名义分布$p(z)$）概率密度分布pdf估算一个期望值时，IS使得我们可以从另一个不同的概率分布（建议分布）中抽样，然后将这些样本对名义分布求期望

$q_\phi(z|x)$表示建议分布，参数$\phi$由参数为$\phi\in\Phi$的神经网络确定，我们可以得到：
$$
\begin{split}
p_\theta(x) 
&= \int p(\hat z)p_\theta(x|\hat z)d\hat z
\\&=E_{p(z)}[p_\theta(x|z)]
\\&=E_{p(z)}[\cfrac{q_\phi(z|x)}{q_\phi(z|x)}p_\theta(x|z)]
\\&=E_{q_\phi(z|x)}[\cfrac{p_\theta(x|z)p(z)}{q_\phi(z|x)}]
\end{split}
$$


PS😕

- [ ] 这里有一个问题还没有理清楚

其实还有另一种后验概率的表达方式

如果说上面表达的z是隐变量，那么这里需要求解的$\theta$怎么表达
$$
p(\theta|x,t) = \cfrac{p(t|x,\theta)p(\theta)}{\int p(t|x,\theta)p(\theta)d\theta}
$$
MLE的角度，基于观测数据t，得到似然函数$\mathbb L(\theta|t,x)$最大时$\theta$的取值，对于回归问题，t是概率分布的自变量（自变量带入**由x和$\theta$共同确定**的概率分布得到概率的值），对于分类问题，t是是否正确分类的标记，如果是正确分类，则取正确分类所对应的概率代入似然函数中。

- [ ] 然而对比本文对隐变量Z的设定，$\theta$瞬间失去了在原本体系中的位置，先验$p(z)$和先验$p(\theta)$各自该如何自处。本这二者合并为z进行理解或许可行,$z = \theta+pre\_z$
- [ ] 当然了，还有EM算法，自难相忘



### 变分推理 vs MCMC

不管是变分推理还是MCMC方法，最终本质上都是解决一个优化问题：
$$
q_\phi^*(z|x)={arg\ min}_{q_\phi(z|x)\in D}KL({q_\phi(z|x)}||{p_\theta(z|x)})
$$
VI通过优化来近似概率分布，广泛应用于估计**使用贝叶斯推理估计后验概率分布时**出现的难以计算的概率,比如上面提到的$p(x)$。

通常我们也用MCMC方法来近似，但是当面对大数据集或者复杂模型，MCMC需要采样大量的样本来进行可靠的估计，但这在效率上是不可接受的。

MCMC产生与目标分布中渐进的精确样本，适合于精确推理，但是计算量大，计算缓慢；VI计算更快，在探索多场景和大数据集下更实用。

### 变分推理

> 定义：从关于隐变量的分布族中找到一个与真实分布差异最小的分布，这个分布的参数用𝝓来标记，即通过寻求最可能接近真实分布的近似分布$p$来逼近真实分布$q$。

- [ ] 那么，当需要利用**Variational Inference**使用变分分布（variational distribution）估计后验概率时，我们都需要做哪些工作？下面这些内容是我们考虑的，后面涉及时会进行介绍：
      - [ ] ELBO(Evidence lower bound)，这是我们最终需要优化的目标函数。
      - [ ] Variational family，定义中提及的关于隐变量的分布族。
      - [ ] Optimization algorithm : Coordinate ascent mean-field variational inference (CAVI)



KL divergence:

我们假设p是一个确定性分布，q分布是待估计分布，

考虑下面两式的不同含义
$$
q^* = argmin_{q}KL(p||q)
$$

$$
q^* = argmin_{q}KL(q||p)
$$



1. q近似p多用的信息量
2. p近似q

<!--由AE的论述转到VAE的论述-->

<!--同时分解说明Encoder，以CNN为例-->

<!--通过GAN来在MNIST数据集上的训练，来说明先验$p(z)$,通过G网络生成$p_{\theta}(x|z)$,-->

定义关于参数$\theta$的后验概率分布：
$$
\begin{split} p(\theta|X,Y,\alpha) &= \cfrac{p(\theta|\alpha)p(Y|X,\theta)}{\int_{\hat \theta}p(\hat\theta|\alpha)p(Y|X,\hat \theta)d\hat \theta}\\&=\cfrac{p(\theta,Y|X,\alpha)}{p(Y|X,\alpha)}\end{split}
$$
其中$\alpha$是$\theta$的先验分布,同时记真实分布为$q(\theta|\phi)$,则使用KL Divergence来衡量近似分布和真实后验概率分布的话，可以表达为：
$$
\begin{split} D_{KL}(q(\theta|\phi)||p(\theta|X,Y,\alpha) )&=E_{q(\theta|\phi)}[log\cfrac{q(\theta|\phi)}{p(\theta|X,Y,\alpha) }]\\&= F(D,\phi)+log\ p(Y|X,\alpha)\end{split}
$$

- [ ] $F(D,\phi)$,变分自由能量的意义是什么

许多概率模型由未归一化的概率分布$\hat p(x;\theta)$定义，必须通过除以配分函数来归一化$\hat p$,以获得一个有效的概率分布。

> 配分函数：是未归一化概率所有状态的积分$\int\hat p(x)dx$（连续变量）或求和$\sum_x\hat p(x)$（离散变量）,而对于很多有趣的模型来说，以上积分或求和难以计算。

通过最大似然学习无向模型特别困难的原因在于配分函数的计算依赖于参数。

#### MCMC

> 构建一个收敛到目标分布的估计序列

当无法精确计算和或者积分（例如，和具有指数数量个项，且无法被精确简化）时，通常可以使用MCMC采样来近似它。这种想法把和或者积分视作某分布下的期望，然后通过估计对应的平均值来近似这个期望。令$s=\sum_x p(x)f(x)=E_p[f(x)]$或者$s=\int p(x)f(x)dx=E_p[f(x)]$为我们需要估计的和或者积分，写成期望的形式，p是一个关于随机变量x的概率分布（求和时）或者概率密度函数（求积分时）。

两个阶段

1. 磨合过程，从运行马尔可夫链开始到其达到均衡分布的过程
2. 从均衡分布中抽取样本序列

通常无法通过表达状态序列，转移矩阵，转移矩阵特征值来判断马尔可夫链是否已经混合成功，只能运行一段足够长的时间并通过启发式的方法判断是否混合成功（包括手动检查样本或者衡量前后样本间的相关性）。

- [ ] MCMC只适用于小规模问题，variational Bayes and expectation propagation的出现使得MCMC可以应用于较大规模问题

收敛：

MCMC的理论可以证明，经过一定次数的迭代之后，本方法一定会收敛的。在一定的迭代次数后所得到的稳定分布会十分接近目标寻求的联合分布。

Burn-in：很明显的最初的一些迭代得到的分布会和目标的后验分布差距很远，因而前N轮的迭代基本是可以直接剔除的。

##### 几种常用的MCMC采样方式

重要采样：

目前的理解是针对复杂的求和或者积分分解出恰当的px和fx，学习的最好的方式是有一个典型的重要采样的例子。



Gibbs采样：

Gibbs采样是最常用的MCMC采样方法

- [ ] 对每一个随机变量产生一个后验条件分布
- [ ] 从目标后验联合分布中模拟后验样本，对每一个随机变量从其它变量固定为当前值的后验条件分布中重复地采样

 Gibbs采样的局限性：

1. 即使得到完全的后验联合密度函数，很多情况下很是难以得到每一个随机变量的条件分布概率；
2. 再退一步，即使得到每一个变量的后验条件分布，可能也不是某种已知的分布形式，无法从中直接进行采样；
3. GIbbs采样可能不适合某些应用场景，效率非常低。



**reference**

- [1][https://am207.github.io/2017/wiki/VI.html#elbo---evidence-lower-bound---objective-function-to-be-optimized](https://am207.github.io/2017/wiki/VI.html#elbo---evidence-lower-bound---objective-function-to-be-optimized)
- [2][http://akosiorek.github.io/ml/2018/03/14/what_is_wrong_with_vaes.html](http://akosiorek.github.io/ml/2018/03/14/what_is_wrong_with_vaes.html)
- [3][https://www.jeremyjordan.me/variational-autoencoders/](https://www.jeremyjordan.me/variational-autoencoders/)





待参阅的资料：

blei的LDA论文的附录，其中便是用mean-field做变分推断的，并且提供了一个完整的LDA变分EM的算法推导。Latent Dirichlet Allocation （LDA）- David M.Blei



EM讲义及EM实现

变分推断

变分EM（推导ppt）