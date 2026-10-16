# 单元03：梯度下降——理解笔记

归档日期：2026-10-16。来源：本聊天单元03的学生作答、修订、手算和闭卷评估。归档日期不代表重新学习或新的结课日期。

本笔记整理学生已经表达或通过计算验证的理解，并把教师补充与学生作答边界分开。课程范围为 markdown/week1.md 的2.5—2.7，只讨论梯度下降及其在线性回归中的作用，不进入多变量线性回归。以下数值和材料情境均为 synthetic data（模拟数据），不是实际实验结果。

## 1. 我已经形成的核心理解

| 核心概念 | 学生最终理解 | 需要保持的精确边界 |
|---|---|---|
| cost | 衡量当前参数组合的预测误差，可理解为当前位置的“高度”；越小通常越好。 | cost只给出当前高度，本身不告诉参数下一步往哪里移动。 |
| gradient | 表示cost关于参数的局部坡度。 | gradient本身指向cost增长最快的方向；负梯度方向才是下降方向。 |
| 为什么需要优化算法 | 只知道某一点的cost，不能证明该点是最低点；梯度下降负责逐步寻找更低cost的参数。 | 代价函数定义“什么叫好”，优化算法负责“怎样寻找更好的参数”。 |
| derivative | 一维情况下，导数表示参数轻微变化时cost怎样变化。 | 导数为负表示局部增大参数会降低cost；导数为正表示局部增大参数会升高cost。 |
| partial derivative | 有多个参数时，只改变其中一个参数并暂时保持其他参数不变，观察cost的变化率。 | 各偏导数组成gradient向量。 |
| learning rate | 学习率alpha控制每次参数更新的步长。 | alpha是人为设定的hyperparameter（超参数），不是模型从本次更新中学习的theta参数。 |
| 同步更新 | theta_0和theta_1的梯度必须从同一个旧参数位置计算，再一起更新。 | 若先更新theta_0再用新theta_0计算g_1，两项梯度就不属于同一个位置。 |
| 收敛 | gradient接近0、参数更新量接近0，且cost变化已经很小。 | 是更新量delta theta接近0，不是theta本身必须趋于0；收敛时cost也不必等于0。 |
| 模型与优化器 | 线性回归规定预测函数的形式，梯度下降负责寻找其中的参数。 | 梯度下降不改变线性模型的形式，也不只用于线性回归。 |

## 2. 数学符号与底层逻辑

本单元中的主要符号：

- x：一个样本的feature（输入特征），材料案例中为ZrB₂含量。
- y：target（目标值），材料案例中为UTS。
- y_hat：prediction（模型预测值）。
- theta_0、theta_1：线性回归模型需要学习的参数。
- J：整个训练集的cost（代价函数）。
- gradient：J对参数的导数或偏导数组成的向量。
- alpha：learning rate（学习率），控制更新步长的超参数。
- m：训练样本数。

单变量线性回归模型为：

$$
\hat y=\theta_0+\theta_1x
$$

平方误差代价函数为：

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}
\left(\hat y^{(i)}-y^{(i)}\right)^2
$$

梯度下降更新公式为：

$$
\theta_j^{new}=\theta_j^{old}-\alpha
\frac{\partial J}{\partial\theta_j}
$$

减号并不表示theta一定减小，而是表示沿负梯度方向移动：

- gradient > 0：更新量为负，参数减小；
- gradient < 0：更新量为正，参数增大。

线性回归的批量梯度为：

$$
g_0=\frac{1}{m}\sum_{i=1}^{m}
\left(\hat y^{(i)}-y^{(i)}\right)
$$

$$
g_1=\frac{1}{m}\sum_{i=1}^{m}
\left(\hat y^{(i)}-y^{(i)}\right)x^{(i)}
$$

这里的batch gradient descent表示每次更新都使用全部训练样本，而不是只看一条记录。

## 3. 为什么梯度中会出现 y_hat - y

学生曾追问：“梯度应该是求偏导，为什么写成y_hat - y？”

对单个样本令：

$$
e=\hat y-y=\theta_0+\theta_1x-y,\qquad J=\frac12e^2
$$

由链式法则：

$$
\frac{\partial J}{\partial\theta_0}
=e\frac{\partial e}{\partial\theta_0}
=e
=\hat y-y
$$

$$
\frac{\partial J}{\partial\theta_1}
=e\frac{\partial e}{\partial\theta_1}
=ex
=(\hat y-y)x
$$

因此误差y_hat - y不是gradient的定义，而是在当前平方误差和线性模型下求偏导后得到的结果。学生随后在x = 2、y = 5、theta_0 = theta_1 = 1时正确算出y_hat = 3、error = -2、g_0 = -2、g_1 = -4。

## 4. 一次同步更新的完整手算

以下为synthetic data：

$$
x=2,\quad y=5,\quad
\theta_0=1,\quad\theta_1=1,\quad\alpha=0.1
$$

学生闭卷计算：

$$
\hat y=1+1\times2=3
$$

$$
error=3-5=-2,\qquad
J=\frac12(-2)^2=2
$$

$$
g_0=-2,\qquad g_1=-2\times2=-4
$$

先用旧参数算完两个梯度，再同步更新：

$$
\theta_0^{new}=1-0.1(-2)=1.2
$$

$$
\theta_1^{new}=1-0.1(-4)=1.4
$$

新预测和新cost为：

$$
\hat y_{new}=1.2+1.4\times2=4
$$

$$
J_{new}=\frac12(4-5)^2=0.5
$$

本步中cost从2降到0.5，说明这个更新方向有效；一次下降不等于整个训练过程已经收敛。

## 5. 连续更新与收敛直觉

在另一条synthetic data上，x = 1、y = 3、初始theta_0 = theta_1 = 0、alpha = 0.1：

| 状态 | theta_0 | theta_1 | y_hat | error | cost |
|---|---:|---:|---:|---:|---:|
| 更新前 | 0 | 0 | 0 | -3 | 4.5 |
| 第1步后 | 0.3 | 0.3 | 0.6 | -2.4 | 2.88 |
| 第2步后 | 0.54 | 0.54 | 1.08 | -1.92 | 1.8432 |

这些结果由学生手算完成并经对话核对。它们展示cost逐步下降，但只有两步，不能据此声称算法已经收敛。

合理的收敛判断可结合：

- 相邻两次cost变化很小；
- gradient的大小接近0；
- 参数更新量的大小接近0；
- 同时设置最大迭代次数，防止无限循环。

真实数据含噪声或模型不能完全拟合时，最小cost通常大于0。

## 6. Python实现及证据边界

学生在对话中独立写出的单样本教学函数为：

~~~python
def gradient_descent(x, y, theta_0, theta_1, alpha, num_steps):
    for _ in range(num_steps):
        y_hat = theta_0 + theta_1 * x
        error = y_hat - y

        gradient_0 = error
        gradient_1 = error * x

        new_theta_0 = theta_0 - alpha * gradient_0
        new_theta_1 = theta_1 - alpha * gradient_1

        theta_0 = new_theta_0
        theta_1 = new_theta_1

    return theta_0, theta_1
~~~

for _ in range(num_steps)表示循环执行num_steps次；下划线表示当前循环编号不会在循环体中使用。num_steps = 5时，下划线依次接收0、1、2、3、4，循环体执行5次。

这段函数体现了同步更新，但只适用于单个标量样本。它没有对多个训练样本求平均，也没有保存cost历史、设置收敛阈值或检测发散。本聊天没有真实Python运行日志，因此只能记录为“独立写出并经人工检查”，不能声称运行测试已经通过。

## 7. 材料科学迁移

假设使用单变量模型研究：

$$
x=\text{ZrB}_2\text{含量}
\quad\longrightarrow\quad
y=\text{7085Al复合材料UTS}
$$

梯度下降调整的是theta_0和theta_1，使训练数据上的cost降低；它不会改变ZrB₂含量，也不会自动验证材料强化机制。

训练后theta_1 > 0，只能说明在当前数据范围和指定线性模型中，ZrB₂含量与预测UTS呈正向关系。它不能单独证明增加ZrB₂含量必然导致真实UTS提高。热处理参数、合金成分、颗粒尺寸与团聚、孔隙率、界面状态等都可能影响UTS。

## 8. 可能失效或误导的条件

教师补充的检查边界：

- alpha太小：通常可以下降，但收敛很慢。
- alpha太大：可能跨过最低点并振荡，甚至使cost持续增大而发散。
- 单变量直线不能表达明显的非线性、最优含量区间或多因素交互。
- 训练范围外的预测属于外推，梯度下降收敛也不保证外推可靠。
- cost下降只说明优化过程改善了指定目标，不证明模型具有因果解释或良好泛化能力。
- 特征尺度差异会影响梯度下降效率；本单元尚未学习多变量特征缩放。
- 对当前平方误差线性回归，cost是凸函数；其他复杂模型的优化不一定具有相同性质。

## 9. 单元结论

单元状态为已完成。闭卷评估中，梯度直觉、导数与偏导、更新公式、学习率、同步更新、手工计算、收敛判断、Python简单实现、材料迁移以及模型与优化器的关系均通过。

当前仍需巩固：严格区分gradient、负梯度方向与实际更新量；提高Python循环和函数语法熟练度；以后独立实现多样本batch gradient descent并实际运行验证。

关联：[练习](../exercises/unit03-gradient-descent.md)｜[错题](../mistakes/unit03-gradient-update-sign.md)｜[测验](../quizzes/unit03-gradient-descent.md)。
