[用机器学习检测异常点击流](https://blog.csdn.net/mergerly/article/details/77985089)

Definition : **Isolation Tree**. Let $T$ be a node of an isolation tree. $T$ is either an external-node with no child, or an internal-node with one test and exactly two daughter nodes $(T_l,T_r)$. A test consists of an attribute q and a split value p such that the test $q < p$ divides data points into $T_l$ and $T_r$. Given a sample of data $X = \begin{Bmatrix} x_1,& ..& ,..&, x_n \end{Bmatrix}$ of $n$ instances from a d-variate distribution, to build an isolation tree (iTree), we recursively divide $X$ by randomly selecting an attribute $q$ and a split value $p$, until either: (i) the tree reaches a height limit, (ii) $|X| = 1$ or (iii) all data in $X$ have the same values. An iTree is a proper binary tree, where each node in the tree has exactly zero or two daughter nodes. Assuming all instances are distinct, each instance is isolated to an external node when an iTree is fully grown, in which case the number of external nodes is n and the number of internal nodes is $n − 1$; the total number of nodes of an iTrees is $2n − 1$; and thus the memory requirement is bounded and only grows linearly with $n$.

The task of anomaly detection is to provide a ranking that reflects the degree of anomaly. Thus, one way to detect anomalies is to sort data points according to their path lengths or anomaly scores; and anomalies are points that are ranked at the top of the list. We define path length and anomaly score as follows

Definition :

- **Path Length** $h(x)$ of a point $x$ is measured by the number of edges x traverses an iTree from the root node until the traversal is terminated at an external node.

- **An anomaly score** is required for any anomaly detection method. The difficulty in deriving such a score from $h(x)$ is that while the maximum possible height of iTree grows

in the order of n, the average height grows in the order of $log n$ [7]. Normalization of $h(x)$ by any of the above terms is either not bounded or cannot be directly compared.

Given a data set of n instances, Section 10.3.3 of [9] gives the average path length of unsuccessful search in BST as:
$$c(n) = 2H(n − 1) − (2(n − 1)/n),$$

where $H(i)$ is the harmonic number and it can be estimated by $ln(i) + 0.5772156649$ (Euler’s constant). As $c(n)$ is the average of $h(x)$ given n, we use it to normalise $h(x)$. The anomaly score s of an instance $x$ is defined as:

$$s(x, n) = 2^{- \frac{E(h(x))}{c(n)}  } ,$$

where $E(h(x))$ is the average of $h(x)$ from a collection of isolation trees. In Equation (2):
• when $E(h(x)) \to c(n), s \to 0.5$;
• when $E(h(x)) \to 0, s \to 1$;
• and when $E(h(x)) \to n − 1, s \to 0$.

$s$ is monotonic to $h(x)$. Figure 2  illustrates the relationship between $E(h(x))$ and s, and the following  conditions applied where $0 < s \leqslant 1$ for $0 < h(x) \leqslant n − 1$. Using the anomaly score s, we are able to make the following  assessment:

• (a) if instances return s very close to 1, then they are definitely anomalies,
• (b) if instances have s much smaller than 0.5, then they are quite safe to be regarded as normal instances, and
• (c) if all the instances return $s \approx 0.5$, then the entire sample does not really have any distinct anomaly.

A contour of anomaly score can be produced by passing a lattice sample through a collection of isolation trees, facilitating a detailed analysis of the detection result. Figure 3 shows an example of such a contour, allowing a user to visualise and identify anomalies in the instance space. Using the contour, we can clearly identify three points, where $s \geq 0.6$, which are potential anomalies.

iForest 适用与连续数据（Continuous numerical data）的异常检测，将异常定义为“容易被孤立的离群点 (more likely to be separated)”——可以理解为分布稀疏且离密度高的群体较远的点。用统计学来解释，在数据空间里面，分布稀疏的区域表示数据发生在此区域的概率很低，因而可以认为落在这些区域里的数据是异常的。

iForest属于Non-parametric和unsupervised的方法，即不用定义数学模型也不需要有标记的训练。对于如何查找哪些点是否容易被孤立（isolated），iForest使用了一套非常高效的策略。假设我们用一个随机超平面来切割（split）数据空间（data space）, 切一次可以生成两个子空间（想象拿刀切蛋糕一分为二）。之后我们再继续用一个随机超平面来切割每个子空间，循环下去，直到每子空间里面只有一个数据点为止。直观上来讲，我们可以发现那些密度很高的簇是可以被切很多次才会停止切割，但是那些密度很低的点很容易很早的就停到一个子空间了。上图里面黑色的点就很容易被切几次就停到一个子空间，而白色点聚集的地方可以切很多次才停止。

怎么来切这个数据空间是iForest的设计核心思想，本文仅介绍最基本的方法。由于切割是随机的，所以需要用ensemble的方法来得到一个收敛值（蒙特卡洛方法），即反复从头开始切，然后平均每次切的结果。iForest 由t个iTree（Isolation Tree）孤立树 组成，每个iTree是一个二叉树结构，一种最基础方法的其实现步骤如下：

1. 从训练数据中随机选择Ψ个点样本点作为subsample，放入树的根节点。

2. 随机指定一个维度（attribute），在当前节点数据中随机产生一个切割点p——切割点产生于当前节点数据中指定维度的最大值和最小值之间。

3. 以此切割点生成了一个超平面，然后将当前节点数据空间划分为2个子空间：把指定维度里小于p的数据放在当前节点的左孩子，把大于等于p的数据放在当前节点的右孩子。

4. 在孩子节点中递归步骤2和3，不断构造新的孩子节点，直到 孩子节点中只有一个数据（无法再继续切割） 或 孩子节点已到达限定高度 。

获得t个iTree之后，iForest 训练就结束，然后我们可以用生成的iForest来评估测试数据了。对于一个训练数据x，我们令其遍历每一棵iTree，然后计算x最终落在每个树第几层（x在树的高度）。然后我们可以得出x在每棵树的高度平均值，即 the average path length over t iTrees。*值得注意的是，如果x落在一个节点中含多个训练数据，可以使用一个公式来修正x的高度计算，详细公式推导见原论文。

获得每个测试数据的average path length后，我们可以设置一个阈值（边界值），average path length 低于此阈值的测试数据即为异常。也就是说 “iForest identifies anomalies as instances having the shortest average path lengths in a dataset ”(异常在这些树中只有很短的平均高度). *值得注意的是，论文中对树的高度做了归一化，并得出一个0到1的数值，即越短的高度越接近1（异常的可能性越高）。
