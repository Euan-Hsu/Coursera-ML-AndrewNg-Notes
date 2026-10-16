# 单元03：梯度下降——练习档案

归档日期：2026-10-16。来源：本聊天实际完成的梯度方向判断、手算更新和Python代码作答。

全部数值与材料情境均为 synthetic data（模拟数据），不是实验结果。验证方式为对话作答与人工算值核对；没有Python实际运行日志，不把人工核对写成“程序运行通过”。

## 1. 单参数更新表达式

学生最终写出：

~~~python
theta_new = theta - alpha * gradient
~~~

在theta = 4、alpha = 0.5、gradient = -2时：

~~~text
alpha * gradient = -1
delta_theta = -alpha * gradient = +1
theta_new = 4 + 1 = 5
~~~

学生当时把delta_theta写成-1，但把theta_new写成5，最终参数值正确而中间更新量符号不一致。后续闭卷终检中，gradient = -4、alpha = 0.1时，学生正确回答delta_theta = +0.4，参数应增大。

结论：更新表达式和最终方向已经掌握，但应继续用“gradient → alpha乘gradient → 前置负号 → theta_new”逐步检查。

## 2. 引导练习：连续两步更新

以下为synthetic data：

~~~text
x = 1
y = 3
theta_0 = 0
theta_1 = 0
alpha = 0.1
~~~

单样本计算规则：

~~~text
y_hat = theta_0 + theta_1 * x
error = y_hat - y
gradient_0 = error
gradient_1 = error * x
cost = 0.5 * error ** 2
~~~

学生经过教学过程中的计算与纠错后完成：

| 状态 | theta_0 | theta_1 | y_hat | error | gradient_0 | gradient_1 | cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| 更新前 | 0 | 0 | 0 | -3 | -3 | -3 | 4.5 |
| 第1步后 | 0.3 | 0.3 | 0.6 | -2.4 | -2.4 | -2.4 | 2.88 |
| 第2步后 | 0.54 | 0.54 | 1.08 | -1.92 | — | — | 1.8432 |

人工核对显示cost连续下降。这里只完成两步，不能据此断言已经收敛。

## 3. 闭卷练习：一次完整同步更新

以下为synthetic data：

~~~text
x = 2
y = 5
theta_0 = 1
theta_1 = 1
alpha = 0.1
~~~

题目要求依次计算：

1. 更新前的y_hat、error和cost；
2. gradient_0和gradient_1；
3. 同步更新后的theta_0和theta_1；
4. 新参数下的y_hat和cost。

学生一次回答：

~~~text
y_hat = 3
error = -2
cost = 2

gradient_0 = -2
gradient_1 = -4

new_theta_0 = 1.2
new_theta_1 = 1.4

new_y_hat = 4
new_cost = 0.5
~~~

评价：闭卷首次完整作答正确。两个gradient都由同一组旧参数计算，更新符合同步要求；cost由2降至0.5。

## 4. 单样本梯度下降函数

学生在教学阶段独立提交的函数：

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

人工代入x = 1、y = 3、theta_0 = theta_1 = 0、alpha = 0.1、num_steps = 2，得到theta_0 = theta_1 = 0.54，与手算一致。

边界说明：

- 这是单样本、标量输入的教学版本。
- gradient_0 = error和gradient_1 = error * x只对应m = 1的情况。
- 它没有实现多个训练样本的batch gradient descent。
- 没有保存cost历史、收敛阈值或发散检测。
- 代码在聊天中书写并经人工检查，没有实际执行日志。

## 5. Python循环语法调试

闭卷复写函数时，学生写成：

~~~python
for _ in range(num_steps):““这行不懂1”“
~~~

### 问题

中文说明和智能引号直接放在冒号后面，不是合法Python语法；若原样执行会产生SyntaxError。

### 教师提示后的正确写法

~~~python
for _ in range(num_steps):  # 重复执行 num_steps 次
~~~

含义：

- range(num_steps)产生0到num_steps - 1；
- for使缩进代码重复执行；
- 下划线接收循环编号，但表示该编号不会被使用；
- 冒号后面的缩进代码属于循环体。

教师追问num_steps = 5时的下划线取值，学生答“0、1、2、3、4”，正确；教师补充循环体一共执行5次。

证据边界：学生理解了循环范围，但没有再次提交一份去掉智能引号的完整函数，也没有实际运行修正版。

## 6. 同步更新代码审查

题目给出：

~~~python
g_0 = gradient_0(theta_0, theta_1)
theta_0 = theta_0 - alpha * g_0

g_1 = gradient_1(theta_0, theta_1)
theta_1 = theta_1 - alpha * g_1
~~~

学生指出：“g_1应该对旧的参数theta_0和theta_1，应该把theta_0的计算放到g_1下面。”

评价：正确识别g_1错误使用了已经更新的theta_0。更精确的更新顺序是先用旧参数计算g_0和g_1，再计算两个new_theta，最后统一赋值。

## 7. 实际完成与未验证部分

| 项目 | 证据与状态 |
|---|---|
| 单参数更新公式 | 经符号纠错后通过 |
| 两步更新手算 | 引导与纠错后完成 |
| 一次完整同步更新 | 闭卷首次完整正确 |
| 同步更新代码审查 | 首次正确指出问题 |
| 单样本函数主体 | 学生独立写出，人工检查正确 |
| for/range语义 | 教师解释后理解 |
| 干净语法版本再次提交 | 未完成 |
| Python实际运行 | 无日志，未验证 |
| 多样本batch gradient descent | 尚未独立实现 |
| NumPy向量化实现 | 本单元未学习 |

后续练习重点：实际运行单样本函数并记录cost；再独立实现多样本平均梯度。在完成之前，不把当前函数表述为完整的批量梯度下降实现。

关联：[理解笔记](../notes/unit03-gradient-descent.md)｜[错题](../mistakes/unit03-gradient-update-sign.md)｜[测验](../quizzes/unit03-gradient-descent.md)。
