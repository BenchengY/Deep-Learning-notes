# Deep Learning 之 参数初始化


本文仅对常见的参数初始化方法进行总结（大部分内容来自[deep learning](http://www.deeplearningbook.org/)一书），原理性的问题不进行过多的探讨。





## Deep Learning中参数初始化十分重要，一般来说有以下这些原因：


	1.初始点的选取，有时候能够决定算法是否收敛
	
	2.当收敛时，初始点可以决定学习收敛的多快，是否收敛到一个代价高或低的点
	
	3.初始化也可以影响泛化
	
	4.初始参数需要破坏不同神经元间的“**对称性**”，因为如果初始化成全0或者一样的值，那么整个神经网络就会变成确定性的算法，具有相同输入的神经元前向传播和梯度更新就都变得一模一样的。（尤其是在cnn中，对于不同channel，具有相同输入的情况下）




**初始化的参数主要有：w权重，bias偏置，下面分别对这两类参数进行讨论**







## 一.w权重：


先上一些结论：
	
	①.在实践中，我们经常将权重初始化为高斯分布，或者均匀分布。高斯分布或者均匀分布的选择并没有太大的区别。但是初始化分布的**大小（scale）**对优化结果和泛化能力有极大的影响。
	
	②对于relu的激活函数推荐使用**【2】或者【3】**的方式






## 1.1均匀分布（uniform initialization）



>1）一种初始化m个输入和n个输出的全连接层的启发式的方法是从分布![这里写图片描述](http://img.blog.csdn.net/20170426092142263?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQlZMMTAxMDExMTE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
中采样权重。

>2）而在【1】中建议采用标准初始化（**normalized initialization**）：
<center>![这里写图片描述](http://img.blog.csdn.net/20170426092830727?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQlZMMTAxMDExMTE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)</center>
这种初始化方式使其具有**相同的激活方差和相同的梯度方差。**




## 1.2 高斯分布（normal initialization）

对于高斯分布，需要设置的参数是均值（mean）和标准差（stddev）。一般情况下，均值均设为0，针对方差不同，有以下几种方式：
>1）设置stddev=0.01或者0.001（一个小的常数）带来的问题是，可能过小而导致前向传播时信号丢失

>2）**Xavier Initialization** (m为输入个数，n为输出个数)
>
 思想是使得输入和输出的方差保持一致（需要注意的是只要满足均值是0，标准差是sqrt(1/m)或sqrt(2/(m+n))分布的都叫做Xavier Initialization。只不过在高斯分布中应用较多，故放在高斯分布的分类下）
 
>  最初的方式是
 <center> **stddev=sqrt（1/m）**</center>

 > 在【2】中推荐设置
 <center> **stddev=sqrt（2/(m+n)**）</center>
    >
3）**MSRA Initialization** 【3】
属于xavier的变种，对于Relu 的激活函数而言，kaiming He 推荐使用 
<center>**stddev=sqrt（2/m）**</center>









## 1.3 Others：

>3正交初始化（**orthogonal initialization**）【4】仔细挑选每一层的非线性缩放或增益（gain）因子g。该初始化方案保证了达到收敛所需的训练迭代总数少于深度。（在LSTM中用到过此方法）

>4.稀疏初始化（**sparse initialization**）【5】
	&emsp; &emsp;主要的思想就是使每一层初始化恰好有k个非0权重。这个想法保持该层的神经元的总数量独立于输入的数目m。使其具有多样性。

>5.使用相同的数据集，用unsupervisor的方式（比如RBM，AE）训练出来的参数初始化supervisor的模型

>6.在相关问题上，用supervisor的方式预训练参数

>7.不相关的task上，supervisor的预训练也可以得到一个比初始化更快收敛的初始值







## 二.对于bias（偏置而言）



一般情况下，偏置总设为0.

可能有如下几个非零情况：
	
	1）如果偏置作为输出单元，初始化偏置以获取正确的输出边缘的统计通常是有利的。
	
	2）有时，我们可能想要选择偏置以避免初始化引起的太大饱和
	
	3）一个单元会控制其他单元能否参与到等式中，比如常见的门操作，在LSTM中将forget_bias=1.0.






     





**参考文献**
【1】Glorot X, Bordes A, Bengio Y. Deep Sparse Rectifier Neural Networks[C]//Aistats. 2011, 15(106): 275.
【2】X. Glorot and Y. Bengio. Understanding the difficulty of training deepfeedforward neural networks. In International Conference on Artificial Intelligence and Statistics, pages 249–256, 2010.
【3】Kaiming He et al., Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classfication
【4】Saxe A M, McClelland J L, Ganguli S. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks[J]. arXiv preprint arXiv:1312.6120, 2013.
【5】Martens, J. (2010). Deep learning via Hessian-free optimization. In ICML’2010, pages 735–742.
