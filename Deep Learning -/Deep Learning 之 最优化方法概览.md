 写在前面本文主要是对Deep Learning一书最优化方法的总结，具体详细的算法，另起博文展开。
 &nbsp;
 >整个优化系列文章列表：
 >
 >[Deep Learning 之 最优化方法](http://blog.csdn.net/BVL10101111/article/details/72614711)
 >
 [ Deep Learning 最优化方法之SGD](http://blog.csdn.net/bvl10101111/article/details/72615436)

>[ Deep Learning 最优化方法之Momentum（动量）](http://blog.csdn.net/bvl10101111/article/details/72615621)
>
>[ Deep Learning 最优化方法之Nesterov(牛顿动量)](http://blog.csdn.net/bvl10101111/article/details/72615961)
>
>[ Deep Learning 最优化方法之AdaGrad](http://blog.csdn.net/bvl10101111/article/details/72616097)
>
>[ Deep Learning 最优化方法之RMSProp](http://blog.csdn.net/bvl10101111/article/details/72616378)
>
>[ Deep Learning 最优化方法之Adam](http://blog.csdn.net/bvl10101111/article/details/72616516)

 
 深度学习中，经常需要用到优化方法，来寻找使得损失函数最小的最优解。
 
 先上一些结论：
> 1.选择哪种优化算法并没有达成共识
> 
 2.具有自适应学习率（以RMSProp 和AdaDelta 为代表）的算法族表现得相当鲁棒，不分伯仲，但没有哪个算法能脱颖而出。
 >
 3.对于当前流行的优化算法包括括SGD、具动量的SGD、RMSProp、具动量的RMSProp、AdaDelta 和Adam而言，**选择哪一个算法似乎主要取决于使用者对算法的熟悉程度（以便调节超参数）**
 >
 4.基本不用二阶近似优化算法



在这里将这些优化方法分为三类，**详见对应的blog：**
 

##一.最基本的优化算法
###[1.1SGD](http://blog.csdn.net/bvl10101111/article/details/72615436)
>SGD实际就是min-batch的实现，为最基础的优化算法，当今大部分优化算法都是以SGD为基础实现的。详见[ Deep Learning 最优化方法之SGD](http://blog.csdn.net/bvl10101111/article/details/72615436)
###[1.2Momentum（动量）](http://blog.csdn.net/bvl10101111/article/details/72615621)
>Momentum引入了动量v，以指数衰减的形式累计历史梯度，以此来解决Hessian矩阵病态问题
	详见[ Deep Learning 最优化方法之Momentum（动量）](http://blog.csdn.net/bvl10101111/article/details/72615621)
###[1.3Nesterov(牛顿动量)](http://blog.csdn.net/bvl10101111/article/details/72615961)
>Nesterov是对Momentum的变种。与Momentum不同的是，Nesterov先更新参数，再计算梯度
	 	详见[ Deep Learning 最优化方法之Nesterov(牛顿动量)](http://blog.csdn.net/bvl10101111/article/details/72615961)


&nbsp;

##二.自适应参数的优化算法
 这类算法最大的特点就是，每个参数有不同的学习率，在整个学习过程中自动适应这些学习率。
###[2.1AdaGrad](http://blog.csdn.net/bvl10101111/article/details/72616097)
>学习率逐参数的除以历史梯度平方和的平方根，使得每个参数的学习率不同
>	 	详见[ Deep Learning 最优化方法之AdaGrad](http://blog.csdn.net/bvl10101111/article/details/72616097)
###[2.2RMSProp](http://blog.csdn.net/bvl10101111/article/details/72616378)
>AdaGrad算法的改进。
	历史梯度平方和--->指数衰减的移动平均,以此丢弃遥远的过去历史。
	 	详见[ Deep Learning 最优化方法之RMSProp](http://blog.csdn.net/bvl10101111/article/details/72616378)
###[2.3Adam](http://blog.csdn.net/bvl10101111/article/details/72616516)
>Adam算法可以看做是修正后的Momentum+RMSProp算法
	详见[ Deep Learning 最优化方法之Adam](http://blog.csdn.net/bvl10101111/article/details/72616516)

&nbsp;
##三.二阶近似的优化算法
二阶近似作为早期处理神经网络的方法，**在此并不另起blog展开细讲。**
###3.1牛顿法
>牛顿法是基于二阶泰勒级数展开在某点附近来近似损失函数的优化方法。主要需要求得Hessian矩阵的逆。如果参数个数是k,则计算你所需的时间是O(k^3)由于在神经网络中参数个数往往是巨大的，因此牛顿法计算法消耗时间巨大。
>
具体更新公式如下：
<center>![这里写图片描述](http://img.blog.csdn.net/20170521212351404?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQlZMMTAxMDExMTE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)</center>

###3.2共轭梯度法
	共轭梯度（CG）是通过迭代下降的共轭方向（conjugate directions）以有效避免Hessian 矩阵求逆计算的方法。
###3.3BFGS
	Broyden-Fletcher-Goldfarb-Shanno（BFGS）算法具有牛顿法的一些优点，但没有牛顿法的计算负担。在这方面，BFGS和CG 很像。然而，BFGS使用了一个更直接的方法近似牛顿更新。用矩阵Mt 近似逆，迭代地低秩更新精度以更好地近似Hessian的逆。
###3.4L-BFGS
	存储受限的BFGS(L-BFGS)通过避免存储完整的Hessian 逆的近似矩阵M，使得BFGS算法的存储代价显著降低。L-BFGS算法使用和BFGS算法相同的方法计算M的近似，但起始假设是M^(t-1) 是单位矩阵，而不是一步一步都要存储近似。
