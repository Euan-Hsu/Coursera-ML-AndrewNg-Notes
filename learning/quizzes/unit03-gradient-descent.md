# 单元03：梯度下降——测验与复测档案

归档日期：2026-10-16。来源：本聊天单元03已经完成的教学问答、闭卷式评估与复测。

“闭卷”是对话中的作答要求，无法核实学生是否查阅材料。以下区分首次回答、提示后修正和教师补充，不补造分数、运行结果或学生未作答的新题。全部数值及材料情境均为 synthetic data（模拟数据），不是实验结果。

## 1. 概念题：为什么已有cost还需要优化算法

题目：只计算J(0)，能否判断theta = 0就是代价最低的位置？为什么还需要梯度下降？

学生回答：“不能，只计算J(0)不能判断这个点就是代价最小的位置。”

评价：首次回答正确。代价函数可以评价给定参数组合，但不会自动给出使其最小的参数；梯度下降负责迭代寻找更低cost的位置。

## 2. 概念题：cost与gradient的区别

题目情境：某点J(theta) = 20、dJ/dtheta = -3。解释cost、gradient以及降低cost的方向。

学生首次回答：“模型的代价函数值为20；梯度为-3；往增大的方向移动。”

评价：方向正确，但前两项只是重复数值，没有解释含义。

教师提示后，学生重新回答：cost说明当前参数组合的预测表现但未必最好；gradient告诉参数怎样变化可以降低cost。

最终评价：提示后通过。教师修订：cost是当前位置的误差高度；gradient是局部坡度并指向cost增长最快的方向，负梯度方向才是下降方向。

## 3. 概念题：偏导数

题目：

1. partial J / partial theta_1表示什么？
2. 计算它时怎样看待theta_0？
3. 若该偏导数等于6，为降低cost，theta_1应怎样移动？

学生回答：

1. “表示cost对参数theta_1的坡度”；
2. “theta_0保持不变”；
3. “减小”。

评价：首次独立全部正确。

## 4. 概念题：学习率过大与过小

题目：alpha太小、太大分别会怎样？哪种情况可能导致无法收敛？

学生首次回答：alpha太小会下降很慢、迭代耗时；alpha太大可能越过最低点；第三问不知道。

教师给出cost序列10 → 16 → 30 → 70并追问。学生回答：“不是（收敛）；太大造成的；学习率太大会无法收敛。”

评价：经情境提示后通过。alpha太大可能振荡或发散；alpha太小通常仍可下降，但速度很慢。

## 5. 概念题：收敛

题目：给出至少两个收敛信号；收敛时cost是否必须等于0？

学生首次回答：“梯度为零，theta变化很小趋向于0。”

评价：部分正确但表述有歧义。教师指出应是gradient接近0、参数更新量delta_theta接近0，不是theta本身趋于0。

进一步情境：数据有波动，最终gradient和delta_theta接近0，但cost大于0。学生回答：“算收敛，cost不是必须等于0，为0即完美预测，但误差总是会有的。”

最终评价：提示后通过。教师补充：真实数据常有误差，但完全匹配且无噪声的synthetic data可能达到cost = 0。

## 6. 数学题：单参数更新与学习率

以下为synthetic data：

$$
\theta=3,\qquad\alpha=0.2,\qquad\frac{dJ}{d\theta}=4
$$

要求计算theta_new，并解释theta、alpha和公式中的减号。

学生首次回答：“theta_new = 1；theta是参数，alpha是变化系数。”

评价：theta角色正确；数值错误，疑似直接减去gradient；alpha解释不完整；没有解释减号。

教师要求先算alpha * gradient。学生最终回答：

~~~text
alpha * gradient = 0.8
theta_new = 2.2
alpha决定模型参数移动的步长
~~~

学生还说明gradient为正时应减去正的更新项。评价：两次提示后通过；alpha应准确称为学习率或超参数。

## 7. 数学题：一次完整同步更新

以下为synthetic data：

$$
x=2,\quad y=5,\quad
\theta_0=1,\quad\theta_1=1,\quad\alpha=0.1
$$

单样本模型：

$$
\hat y=\theta_0+\theta_1x,\qquad
J=\frac12(\hat y-y)^2
$$

梯度：

$$
g_0=\hat y-y,\qquad
g_1=(\hat y-y)x
$$

学生一次回答：

~~~text
更新前：y_hat = 3，error = -2，cost = 2
梯度：g_0 = -2，g_1 = -4
新参数：theta_0 = 1.2，theta_1 = 1.4
更新后：y_hat = 4，cost = 0.5
~~~

评价：闭卷首次完整正确。cost由2下降至0.5，且两个参数使用旧参数位置的梯度同步更新。

## 8. 数学终检：负gradient下的实际更新量

以下为synthetic data：

$$
\frac{dJ}{d\theta}=-4,\qquad\alpha=0.1
$$

学生回答：“0.4；增大；因为梯度为负数。”

评价：delta_theta = +0.4和参数增大方向正确。教师补足理由：gradient < 0表示局部增大theta会降低cost，因此负梯度方向指向theta增大的方向。

## 9. Python题：单参数更新表达式

要求写出一行Python更新表达式。

学生写出：

~~~python
theta_new = theta - alpha * gradient
~~~

评价：表达式正确。此前测试theta = 4、alpha = 0.5、gradient = -2时，学生给出theta_new = 5正确，但曾把delta_theta的符号写错；详见错题档案。

## 10. Python题：单样本多步gradient_descent函数

题目要求：

- 循环num_steps次；
- 计算y_hat、error、两个gradient；
- 同步更新theta_0和theta_1；
- 返回两个参数；
- 不使用NumPy。

学生提交：

~~~python
def gradient_descent(x, y, theta_0, theta_1, alpha, num_steps):
    for _ in range(num_steps):
        y_hat = theta_0 + theta_1 * x
        error = y_hat-y

        gradient_0 = error
        gradient_1 = error * x

        new_theta_0 = theta_0 - alpha * gradient_0
        new_theta_1 = theta_1 - alpha * gradient_1

        theta_0 = new_theta_0
        theta_1 = new_theta_1

    return theta_0, theta_1
~~~

评价：算法主体和同步更新正确。人工核对x = 1、y = 3、初始theta_0 = theta_1 = 0、alpha = 0.1、num_steps = 2，结果为0.54、0.54，与手算一致。

证据边界：没有实际运行日志；函数只处理单个样本，不是多样本batch gradient descent。

## 11. Python题：for与range的理解

闭卷复写时，学生在for行后直接加入中文智能引号说明，形成无效语法：

~~~python
for _ in range(num_steps):““这行不懂1”“
~~~

教师解释for、range、下划线和冒号，并改为注释形式。随后提问num_steps = 5时循环次数及下划线取值。

学生回答：“0、1、2、3、4”。

评价：取值正确，循环共执行5次由教师补充。循环含义解释后理解；学生没有再次提交干净完整版本，也没有运行验证。

## 12. 概念题：同步更新代码审查

题目代码：

~~~python
g_0 = gradient_0(theta_0, theta_1)
theta_0 = theta_0 - alpha * g_0

g_1 = gradient_1(theta_0, theta_1)
theta_1 = theta_1 - alpha * g_1
~~~

学生指出g_1应基于旧theta_0和theta_1，因此theta_0的赋值应放到g_1计算之后。

评价：首次正确识别问题。更完整的做法是先算完g_0、g_1，再计算两个新参数，最后统一赋值。

## 13. 材料科学迁移题

以下是假设性教学情境，不是实验结论：

使用ZrB₂含量x预测7085Al复合材料UTS，梯度下降训练后theta_1 > 0。

问题：

1. theta_1 > 0说明什么？
2. 能否证明增加ZrB₂含量必然导致UTS提高？
3. 梯度下降负责什么，是否改变线性回归形式？

学生回答：

1. “说明ZrB₂含量对UTS_hat的影响呈现正相关”；
2. “不能，因为还有其他因素的影响”；
3. “负责学习获得最优参数使得代价函数最低，它不改变模型形式”。

评价：首次整体通过。教师精确修订：第一项应说在当前数据范围和指定模型中，ZrB₂含量与预测UTS呈正向关系，避免把模型关系称为真实因果影响；梯度下降寻找较低cost的参数，但有限迭代不保证已经达到精确全局最小值。

## 14. 最终综合表述

学生最终总结能够说明：

- cost评价当前参数组合；
- 梯度下降在cost定义后继续优化参数；
- alpha控制步长；
- 参数应从同一旧位置同步更新；
- gradient和参数更新量接近0可作为收敛信号；
- 线性回归是模型，梯度下降是参数优化算法。

教师对总结中的两点作了最后修订：

1. gradient本身是坡度信息，不是优化动作；
2. 更新公式的减号表示沿负梯度方向，不表示theta总是减小。

终检题通过后，单元判定为完成。

## 15. 单元评价

| 评估项目 | 结论与证据边界 |
|---|---|
| 梯度直觉 | 最终通过；cost与gradient表述经提示修正 |
| 导数与偏导数 | 通过；偏导题首次正确 |
| 更新公式及减号 | 经多次符号纠错和终检后通过 |
| 学习率 | 经提示后通过 |
| 同步更新 | 通过；手算与代码审查均正确 |
| 手工计算 | 通过；完整更新题首次全对 |
| 收敛判断 | 经术语澄清后通过 |
| Python简单实现 | 对话代码通过；循环语法仍需练习；未运行验证 |
| 材料科学迁移 | 通过；因果措辞经教师精确修订 |
| 模型与优化器关系 | 通过 |
| 总体掌握程度 | 基本掌握，核心机制已掌握 |
| 单元状态 | 已完成 |

本档案包含实际完成的概念、数学、Python和材料迁移题，不补造额外题目、分数或运行日志。

关联：[理解笔记](../notes/unit03-gradient-descent.md)｜[练习](../exercises/unit03-gradient-descent.md)｜[错题](../mistakes/unit03-gradient-update-sign.md)。
