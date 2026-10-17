# 单元03：梯度下降——错题与易混淆点

首次记录日期：2026-10-09。扩充归档日期：2026-10-16。依据本聊天单元03的实际作答与闭卷复测。

所有数值均为 synthetic data（模拟数据）。以下保留历史错误和修正过程；“通过”表示本单元结束时的即时验证结果，不等于长期不会再犯。

## 1. gradient、负梯度方向与参数更新量的符号混淆

### 错误现象

在下列条件下：

~~~text
theta = 2
alpha = 1
gradient = dJ/dtheta = -5
~~~

学生连续两次把参数更新方向判断为减小，并写出错误结果theta_new = -3。

另一次在theta = 4、alpha = 0.5、gradient = -2时，学生把delta_theta写成-1，却同时写出theta_new = 5。最终参数值与中间更新量的符号互相矛盾。

### 当时的错误理解

把gradient或alpha * gradient直接当成最终参数更新量，漏掉了更新公式最前面的负号；同时混淆了三个不同的量：

~~~text
gradient
-alpha * gradient
theta_new
~~~

### 正确解释

应拆成：

$$
\Delta\theta=-\alpha\cdot gradient
$$

$$
\theta_{new}=\theta_{old}+\Delta\theta
$$

当alpha > 0时：

- gradient > 0，则delta_theta < 0，theta减小；
- gradient < 0，则delta_theta > 0，theta增大。

严格措辞：gradient < 0表示局部增大theta会使J下降；此时-gradient > 0，所以“负梯度方向”指向theta增大的方向。“gradient为负”和“负梯度方向”不是同一句话。

### 如何发现问题

教师把计算拆成：

~~~text
gradient
→ alpha * gradient
→ -alpha * gradient
→ theta_new
~~~

并用局部坡度检查方向。theta = 4那题中，若delta_theta真为-1，则theta_new应为3，不可能同时等于5；这种自相矛盾暴露了符号错误。

### 防止再次发生的检查方法

1. 单独写出gradient的符号。
2. 计算alpha * gradient。
3. 再处理公式前面的减号，得到delta_theta。
4. 用theta_new - theta_old反算，检查是否等于delta_theta。
5. 最后用坡度直觉检查方向，不背诵“有减号所以theta减小”。

补充边界：负梯度只是局部下降方向。alpha过大时，即使方向正确，也可能跨过最低点、振荡或发散。

### 最新验证状态

闭卷末题中，dJ/dtheta = -4、alpha = 0.1，学生正确回答delta_theta = +0.4、theta增大。数值与方向即时复测通过；“增大theta会在局部降低cost”的完整理由由教师补足。长期保持仍需后续无提示检查。

## 2. cost、gradient与优化动作没有严格区分

### 错误现象

在闭卷解释中，学生一度只重复“cost为20、gradient为-3”，没有说明二者含义。后续总结又写“gradient为梯度，可以优化参数提高模型预测能力”。

### 当时的错误理解

把gradient本身说成优化动作，或者把它笼统描述为“告诉theta往哪里走就能下降”，没有区分gradient方向和负gradient方向。

### 正确解释

- cost是当前参数组合在指定代价函数下的误差高度。
- gradient是cost关于参数的局部变化率，并指向cost增大最快的方向。
- 梯度下降算法执行的更新动作是沿负梯度方向移动参数。

gradient不是cost，也不是参数更新本身；它是优化算法使用的信息。

### 如何发现问题

教师追问“cost表示什么、gradient表示什么、二者核心区别是什么”，并用“高度”和“坡度”要求学生重新表述。最终学生能说明cost评价当前参数组合，而gradient提供参数变化信息；教师再补充gradient本身指向上升方向。

### 防止再次发生的检查方法

每次说明时固定使用三句话：

1. cost是高度；
2. gradient是坡度并指向最陡上升；
3. -gradient才是局部最陡下降方向。

### 最新验证状态

最终能区分cost与gradient，但总结措辞仍经教师精确修订。概念通过，表达准确性需继续保持。

## 3. 参数theta与学习率alpha混淆及漏乘alpha

### 错误现象

闭卷题：

~~~text
theta = 3
alpha = 0.2
gradient = 4
~~~

学生首次回答theta_new = 1，并把alpha称为“变化系数”。

### 当时的错误理解

计算时似乎直接用theta - gradient，漏掉alpha；对alpha的角色描述不够明确。

### 正确解释

$$
\alpha\cdot gradient=0.2\times4=0.8
$$

$$
\theta_{new}=3-0.8=2.2
$$

theta是模型从训练数据中调整的parameter（参数）；alpha是人为设定的learning rate（学习率），属于hyperparameter（超参数），控制更新步长。

### 如何发现问题

教师要求先单独计算alpha * gradient，再代入更新公式，并追问alpha决定什么。学生最终回答“0.8、2.2；决定了模型参数移动的步长”。

### 防止再次发生的检查方法

更新前依次标注：

~~~text
模型参数：theta
超参数：alpha
坡度信息：gradient
实际更新量：-alpha * gradient
~~~

检查实际移动量是否同时包含alpha和gradient。

### 最新验证状态

提示后通过。后续手工同步更新和最终符号题均正确。

## 4. 收敛时是更新量接近0，不是参数本身接近0

### 错误现象

学生首次回答收敛条件为：“梯度为零，theta变化很小趋向于0。”

### 当时的错误理解

“theta变化趋近0”表述有歧义，容易被理解为theta本身必须趋于0；同时没有回答收敛时cost是否必须为0。

### 正确解释

常见判断信号包括：

- gradient的大小接近0；
- 参数更新量delta_theta接近0；
- 相邻迭代的cost变化很小。

theta本身可以收敛到任何合适数值，不必接近0。真实数据含噪声或模型不能完全拟合时，最小cost可以大于0。

### 如何发现问题

教师用“最优参数可能是25”区分theta与delta_theta，并给出直线不能穿过所有有波动样本的情境。学生随后回答：“算收敛，cost不是必须等于0，为0即完美预测，但误差总是会有的。”

教师补充边界：真实数据通常存在误差，但无噪声且模型完全匹配的synthetic data也可能达到cost = 0，不能说误差在所有情况下“总是”存在。

### 防止再次发生的检查方法

写收敛条件时明确使用delta_theta或theta_new - theta_old，不只写theta；另问自己：“算法是否停止改善”和“训练误差是否恰好为0”是同一个问题吗？

### 最新验证状态

提示后通过，能说明cost不必为0。

## 5. 本单元错题结论

主要重复错误是更新符号；其余为概念表述或单次计算问题。单元结束时即时复测已通过，但后续学习仍应重点检查：

- gradient与负梯度方向；
- gradient与delta_theta；
- theta与alpha；
- theta本身与delta_theta；
- 局部下降方向与有限步长实际是否下降。

Python闭卷复写时把中文智能引号直接放入for行的问题属于一次语法调试，证据不足以称为反复错误，记录在[练习档案](../exercises/unit03-gradient-descent.md)中。

关联：[理解笔记](../notes/unit03-gradient-descent.md)｜[练习](../exercises/unit03-gradient-descent.md)｜[测验](../quizzes/unit03-gradient-descent.md)。
