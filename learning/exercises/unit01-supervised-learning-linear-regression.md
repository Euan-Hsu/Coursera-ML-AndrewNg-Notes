# 单元01：监督学习与单变量线性回归基础——Python练习档案

归档日期：2026-10-16。来源：本聊天真实题目与学生作答。所有材料参数和数值为synthetic data，仅用于教学。

验证类型：对话中书写代码及人工算值核对；没有实际终端或Notebook运行记录。以下内容不构成运行通过证明。

## 练习1：写预测表达式

题目给定：

~~~python
theta_0 = 500
theta_1 = 40
x = 0.75
~~~

任务：补写一行代码，计算预测UTS并保存到y_hat。

### 学生首次作答

~~~python
y_hat = theta_0 + theta_1 * x
~~~

学生原消息使用ini代码围栏；归档仅将显示标签调整为python，代码表达式不变。

### 思路与评价

学生没有单独提交文字思路；代码正确对应“截距 + 斜率 × 输入”，属于首次独立作答通过。不能据此虚构其编程过程。

## 练习2：定义预测函数

题目：

~~~python
def predict_uts(x, theta_0, theta_1):
    # 补写函数体
~~~

### 学生首次代码

以下只规范化聊天转义和缩进显示，保留原来“赋值但没有return”的代码逻辑：

~~~python
def predict_uts(x, theta_0, theta_1):
    predict_uts = theta_0 + theta_1 * x
~~~

### 问题与提示

- 缺少显式return；函数结束时默认返回None。
- 在函数内部将局部变量命名为predict_uts并非Python语法错误，但容易混淆函数与数值，初学时应避免。
- 教师提示保留函数定义，在缩进后的下一行用return返回预测表达式。
- 学生随后明确说：“我不会写代码”。

聊天中的反斜杠、HTML空格等展示字符未经过实际执行，不能据此声称复现了语法异常。

### 教师随后提供的代码

以下是教师示例，不是学生独立完成的函数：

~~~python
def predict_uts(x, theta_0, theta_1):
    return theta_0 + theta_1 * x
~~~

教师解释了def、函数名、输入、冒号、缩进、return及函数调用。

给出的调用示例：

~~~python
result = predict_uts(0.75, 500, 40)
~~~

学生回答调用结果为“530”，数值判断正确。教师当时显示为530.0；本档案只记录二者数值相同，不将数值判断当作实际运行输出。

### 最终验证状态

| 项目 | 状态 |
|---|---|
| 线性预测表达式 | 首次独立作答正确 |
| 原函数返回值逻辑 | 首次缺少return |
| 完整函数实现 | 教师提供；本单元没有独立重写验证 |
| 读懂调用结果 | 看过教师示例后数值判断正确 |
| 实际运行 | 未提供运行日志 |

## 后续练习方向

本单元结束时需要继续练习函数输入、缩进、return和调用。这里只记录方向，不新建未做过的题目、不伪造新的答案。后续单元已经取得的编程进展应读取最新progress.md，不将单元01的状态永久套用。

关联：[理解笔记](../notes/unit01-supervised-learning-linear-regression.md)｜[错题](../mistakes/unit01-supervised-learning-linear-regression.md)｜[测验](../quizzes/unit01-supervised-learning-linear-regression.md)。
