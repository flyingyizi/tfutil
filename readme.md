

# introduction

目前存在几种不同类型的学习算法。主要的两种类型被我们称之为监督学习和无监督学习。简单来说，监督学习这个想法是指，我们将教计算机如何去完成任务，而在无监督学习中，我们打算让它自己进行学习。此外还有诸如，强化学习和推荐系统等各种术语。这些都是机器学习算法的一员，以后我们都将介绍到，但学习算法最常用两个类型就是监督学习、无监督学习。

NG(吴恩达)建议：如果使用Octave作为编程环境，如果使用Octave作为学习工具，以及作为原型工具，它会让你对学习算法的学习和建原型快上许多。

在下面讨论中，描述问题的约定缩写包括：
- $m$ 代表训练集中实例的数量
- $x$ 代表特征/输入变量
- $y$ 代表目标变量/输出变量
- $(x,y)$ 代表训练集中的实例
- $(x^{(i)},y^{(i)})$ 代表第 个观察实例
- $h$ 代表学习算法的解决方案或函数也称为假设（**hypothesis**）
- $J$ 代表代价函数

模型所预测的值与训练集中实际值之间的差距就是建模误差（modeling error）,通常定义代价函数采用平方误差代价函数：
$$J(\theta)=\frac{1}{2m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2 $$

目标就是找到适合的$\theta$,使得 $minimize(J(\theta)) $

# csvdata package
in this package,provide same help function to read csv data and output to gonum dense


# linear regression package

in this package, implement linear gregress algorithm. 

# logic regression package 

