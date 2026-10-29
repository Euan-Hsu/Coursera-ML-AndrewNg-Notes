# 单元09：学习曲线与K折交叉验证——理解笔记

状态：学习中；2026-10-29启动。已收到单元09概念入口Q0首答，当前提示纠正中；尚未获得正确复答或独立复查证据。

## 已核对的前置证据

本地ML_LEARNING_HANDOFF.md最新入口、GitHub的AGENTS.md、STUDY_ROADMAP.md、learning/progress.md、单元08四类档案及Notebook均已核对，远程基线为1f49dda9c9e1608854bb5194b2e68beb0ff7d22a。

单元08本轮已完成：正式8题通过8/8，其中无追加内容提示5题，提示后3题；学习曲线横轴和用途曾首次回答正确，但没有实际学习曲线代码或图形证据。保留全部历史帮助层级，不重新教学Bias/Variance或Ridge整章，也不把单元08测试集恢复成独立新数据。

## 当前知识点与学生理解

当前只检查“学习曲线的每个点如何得到”。

> 模型的多项式次数和Ridge的alpha都保持固定。要画训练样本数分别为12、24、48时的误差点：你会分别用对应数量的训练样本重新fit，还是先用全部训练样本fit一次，再换不同数量的样本计算误差？请说明理由。

学生首次原答（2026-10-29）：“第二种吧，原因不会”。选择第二种不符合训练样本量学习曲线的目标；理由未说明，不推断误解的具体来源。

教师提示（不是学生自述）：如果先用96条synthetic训练样本fit，再取其中12条计算误差，模型的系数仍是从96条数据学来的；只减少评分样本没有减少训练时的信息。横轴写“训练样本数12”时，应代表模型实际用多少条学习？请据此重新选一次并说明理由。

提示后纠正：尚未收到，待回答。  
独立复查：尚未进行。

当前只观察到一次选择错误，未记为重复错误；本题不计入正式8题。

## 本单元范围与来源

- 课程markdown/week6.md第10.3—10.6节及code/ex5-bias vs variance/。
- 学习曲线的坐标、训练/验证误差、Bias/Variance典型形态及诊断限制，逐步实现并由学生先读图。
- 固定一次划分的波动、K=5轮换、折内Pipeline、KFold/cross_validate、负MAE、每折MAE及均值/标准差。
- GroupKFold仅认识炉次与文献来源独立性原理，不进入完整分组实践。
- 仍使用已学一次线性回归与一个预设多项式Ridge方案。

教师备课修正：课程10.3的“交叉验证集”主要指固定验证集，不是K折；其测试公式存在cv下标笔误。Ex5历史代码分别用各组自身统计量标准化，并以测试误差选lambda，这些步骤不照搬。原课程不改。

## 材料迁移与科学边界

新数据来自synthetic data数学教学设定，非7085Al实验。CV均值不等于真实材料性能，不保证新炉次、域外成分或不同文献来源可靠。上述是教师设定与待学边界，尚未记为学生通过。

## 学习产出

- [Notebook](../exercises/unit09-learning-curves-kfold-cross-validation.ipynb)
- [练习与实验预设](../exercises/unit09-learning-curves-kfold-cross-validation.md)
- [错误与复查](../mistakes/unit09-learning-curves-kfold-cross-validation.md)
- [测验记录](../quizzes/unit09-learning-curves-kfold-cross-validation.md)
