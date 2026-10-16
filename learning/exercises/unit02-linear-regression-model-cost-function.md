# 单元02：单变量线性回归模型与代价函数——练习档案

归档日期：2026-10-16。来源：本聊天实际开展的手算和Python阅读练习。全部材料数值为 synthetic data（模拟数据），不属于真实实验记录。

验证边界：以下仅有对话作答和人工算值核对，没有运行记录。本单元Python函数均由教师给出，学生没有独立提交完整程序；不能称为独立编程完成或运行通过。

## 1. 预测函数：同时处理三个输入

教师给定代码：

~~~python
import numpy as np

def predict(x, theta_0, theta_1):
    return theta_0 + theta_1 * x

x = np.array([0.0, 0.5, 1.0])
y_hat = predict(x, 500, 40)
~~~

含量x按wt.%数值编码，y_hat为MPa。任务：不运行，手算y_hat的三个数值。

学生首次答案：“500,520,540”。

评价：首次作答正确。学生能将同一组参数分别代入三个输入；没有提交文字推理或实际代码执行过程。

教师整理后的数学对应：

~~~text
x = 0.0 → y_hat = 500
x = 0.5 → y_hat = 520
x = 1.0 → y_hat = 540
~~~

## 2. cost函数：数组预测、相减、平方与汇总

教师给定代码：

~~~python
import numpy as np

def predict(x, theta_0, theta_1):
    return theta_0 + theta_1 * x

def compute_cost(x, y, theta_0, theta_1):
    m = len(x)
    y_hat = predict(x, theta_0, theta_1)
    errors = y_hat - y
    squared_errors = errors ** 2
    cost = np.sum(squared_errors) / (2 * m)
    return cost

x = np.array([0.0, 0.5, 1.0])
y = np.array([502.0, 518.0, 541.0])
cost = compute_cost(x, y, 500, 40)
~~~

这里y是模拟目标数组，不是实测实验数据。输入和目标按同一顺序配对。

任务：手算errors、squared_errors和cost。

### 学生首次回答

“1得到每个样本误差2得到每个误差平方3cost=1.5”。

评价：cost正确；前两项说明了含义，但未提供题目要求的数组值。不能记为首次完整回答。

### 提示与补答

教师给出逐项相减框架，要求补全数值：

~~~text
errors = [500 - 502, 520 - 518, 540 - 541]
~~~

学生补答：“[-2，2，-1]、[4，4，1]”。

评价：补答正确。

### 最终核对

~~~text
errors = [-2, 2, -1] MPa
squared_errors = [4, 4, 1] MPa²
sum(squared_errors) = 9 MPa²
m = 3
cost = 9 / 6 = 1.5 MPa²
~~~

每个误差仍是一条记录的结果；cost是三个样本汇总后的一个标量。

## 3. 闭卷代码解释

教师题干：

~~~python
def compute_cost(x, y, theta_0, theta_1):
    m = len(x)
    y_hat = theta_0 + theta_1 * x
    errors = y_hat - y
    squared_errors = errors ** 2
    return squared_errors.sum() / (2 * m)
~~~

学生回答：

1. x、y需要数组：“因为数组能进行数学运算”。
2. errors ** 2：“对误差进行了平方”。
3. 返回值：“返回的是cost数值”。

评价：基本通过。第一项不完整，教师补充数组分别保存多个样本的输入和目标，从而逐项配对运算。第三项正确识别整体标量。

教师补充的实现前提：这里x、y应是同长度、顺序对应的一维NumPy数组，避免Python列表运算或二维/一维混合广播造成与公式不同的计算。本前提是归档解释，不声称学生已经进行shape测试。

## 4. 手算错误与独立复测

闭卷给出误差[-3, 1, 5]，学生先回答J = 7/3。教师要求写出平方、求和、2m后，学生更正为35/6。

后续新题误差[-4, 2, 6]，学生独立回答：

~~~text
sum(e²) = 56
2m = 6
J = 9.3
~~~

核对：56/6约9.33，学生9.3为一位小数近似，数值判断正确。不能将提示后的第一题更正标作首次独立全对。

## 5. 实际完成与未验证部分

| 项目 | 证据与状态 |
|---|---|
| 三个输入的批量预测 | 首次手算正确 |
| 误差与平方数组 | 提示后补全正确 |
| 整体cost | 教学题正确；闭卷先错后改；新题独立复测正确 |
| 读懂函数返回值 | 闭卷正确 |
| 解释数组目的 | 初答笼统，教师补充 |
| 独立函数编写 | 本单元未完成该验证 |
| 运行或调试 | 无日志，不宣称运行通过 |

后续练习方向：独立写预测和cost函数，检查配对数组与返回值；不在此档案中添加未完成的新题或伪造运行结果。

关联：[理解笔记](../notes/unit02-linear-regression-model-cost-function.md)｜[错题](../mistakes/unit02-linear-regression-model-cost-function.md)｜[测验](../quizzes/unit02-linear-regression-model-cost-function.md)。
