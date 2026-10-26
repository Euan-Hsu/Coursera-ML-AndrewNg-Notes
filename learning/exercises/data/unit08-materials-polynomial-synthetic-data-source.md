# Unit 08 synthetic data source

生成日期：2026-10-26。教师准备的数据资产，不是学生实验结果或模型训练输出。

## 数据与科学边界

- 文件：`unit08-materials-polynomial-synthetic.csv`
- 明确类别：**synthetic data**；全部输入和目标按人为规则生成，未使用真实7085Al/ZrB₂实验数据。
- 50行、2列；输入`ZrB2_wt_pct`按wt.%数值编码，目标`UTS_MPa`单位MPa。
- 单输入仅用于学习非线性拟合、过拟合和正则化，忽略真实成分、工艺、组织等因素。
- 曲线峰值由教师生成规则设定，不能解释为真实最佳含量、强化机制、实验窗口或材料性能预测有效性。
- 本次没有拟合模型、预测或计算任何MAE/RMSE，也没有挑选能产生某种误差排序的seed。

## 一次生成协议

实际环境：Python 3.10.11，NumPy 2.2.6，PCG64/default_rng，seed=20261026。

1. `x = np.round(np.linspace(0.0, 2.0, 50), 3)`。
2. `rng = np.random.default_rng(20261026)`，一次取`rng.normal(0, 6, 50)`。
3. `y = np.round(470 + 70*x - 35*x**2 + noise, 2)`。
4. CSV头为`ZrB2_wt_pct,UTS_MPa`，x输出3位小数、y输出2位小数，UTF-8无BOM、LF、文件末尾一个换行。
5. CSV由本地Python在内存一次生成，再写入资产；固定规则和seed后没有重新选数据。
6. 不根据生成规则的已知峰值调整候选或查看测试性能。

CSV SHA256：`fc097879abfae15b6def06b5b89254ba3144aaa9380a4c4ae270fe54dade54ab`

## 预定实验

- 先划分，再进行任何学习型预处理。两次train_test_split均random_state=42，第一次训练60%、留出40%，第二次将留出部分等分验证/测试，30/10/10。
- 同一划分复用于所有候选；保留行索引用于核查。
- 次数：1、2、12；12次Ridge的alpha：0.01、0.1、1、10、100、1000。
- Pipeline：PolynomialFeatures(include_bias=False)、StandardScaler、LinearRegression/Ridge，仅训练fit。
- 所有变换后的统计量来自训练数据；验证/测试和网格由完整Pipeline直接predict原始二维输入。
- y沿用本课程二维列形式(m,1)，绘图按需明确转换，MAE/RMSE单位MPa。
- 依据验证集选择一个方案；测试不画点、不比较候选，最后固定方案后评价一次。
- 预先不承诺高次必定过拟合、某个alpha必定最优或Ridge必定改善；以学生真实运行结果讨论。

## 当前验证范围

教师已成功导入本单元依赖：pandas 2.3.3、scikit-learn 1.7.2、nbformat 5.11.1；这不是学生执行、模型训练或完整流程验证。Notebook学生格初始execution_count为null、outputs为空。

启动：双击同目录`START_UNIT08.cmd`，或在已有JupyterLab打开本单元Notebook。内核使用已有`Unit 07 (.venv)`（同一环境），工作目录应为`learning/exercises/`。环境仅复用既有本地配置，不宣称跨机器复现已经验证。
