# Unit 03：梯度更新中的符号错误

日期：2026-10-09

## 错误现象

已知：

[
	heta=2,qquad alpha=1,qquad rac{dJ}{d	heta}=-5
]

连续两次把负梯度对应的参数更新方向判断为减小，并写出错误结果 (	heta_{mathrm{new}}=-3)。

## 错误理解

混淆了 gradient 本身与参数更新量：

[
Delta	heta=-alpharac{dJ}{d	heta}
]

当 gradient 为负数时，更新量不是负数，而是“负号乘负数”得到的正数。

## 正确解释

按三步计算：

[
g=rac{dJ}{d	heta}=-5
]

[
alpha g=1	imes(-5)=-5
]

[
Delta	heta=-alpha g=-(-5)=+5
]

所以：

[
	heta_{mathrm{new}}=2+5=7
]

负梯度表示增大 (	heta) 可以在局部降低代价，因此参数应向增大的方向更新。

## 防止再次发生的检查方法

1. 先单独写出 gradient 的符号。
2. 再计算 (alpha g)。
3. 最后处理更新公式最前面的减号。
4. 检查更新方向是否与直觉一致：
   - gradient (>0)：参数减小；
   - gradient (<0)：参数增大。
