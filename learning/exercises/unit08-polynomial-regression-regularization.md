# 单元08练习与运行证据

## 当前状态

2026-10-26：概念教学后准备进入实践。尚无学生Python提交、执行或模型结果。正式单元末评估尚未开始，单元未完成。

## 独立资产

- [空白Notebook](unit08-polynomial-regression-regularization.ipynb)：7个学生代码格，初始只有任务注释；execution_count全部null，outputs全部空。
- [synthetic CSV](data/unit08-materials-polynomial-synthetic.csv)：50行、2列，输入ZrB2_wt_pct和目标UTS_MPa。
- [完整来源与协议](data/unit08-materials-polynomial-synthetic-data-source.md)：一次生成规则、固定seed、单位、科学边界和校验哈希。
- 本地启动器：`learning/exercises/START_UNIT08.cmd`，使用既有.venv；不是新环境安装或跨机器复现证明。

本单元不复制或覆盖单元07Notebook。单元07SHA256仍为B466122C36DBBFFA3C1B5BEFBD0A71BAEB3BE1D0D860063D7C2204DFA365F2DF。

## 教师数据准备的真实执行

教师在独立Python进程导入依赖、在内存一次生成模拟数据，再保存为新资产。没有拟合、predict、评分、实际划分或执行学生Notebook。

真实核验结果：

```text
notebook_valid: true
student_code_cells: 7
execution_counts: [null, null, null, null, null, null, null]
stored_outputs: 0
csv_shape: [50, 2]
csv_bytes: 670
csv_sha256: fc097879abfae15b6def06b5b89254ba3144aaa9380a4c4ae270fe54dade54ab
unit07_unchanged: true
```

来源：2026-10-26教师侧验证进程退出码0；nbformat.validate、pandas读取、列名/有限值/唯一输入/范围/校验哈希/空执行记录断言通过。不是学生读取成功、学生shape回答、模型输出或训练流程验收。

环境实际导入：Python3.10.11、NumPy2.2.6、pandas2.3.3、scikit-learn1.7.2、nbformat5.11.1。

## 预定实践边界

50行固定60/20/20，两次拆分random_state=42，30/10/10；所有候选复用同一分组。次数1/2/12；12次Ridge的alpha为0.01、0.1、1、10、100、1000。没有为了预期误差排序筛选seed。

Pipeline组合多项式转换(include_bias=False)、标准化和回归；仅训练fit。传给完整Pipeline.predict的是原始二维输入，不重复手动展开/标准化。y及预测沿用二维列约定。保留测试直到最终固定一个方案，不画测试点、不给每个候选评分。

全部材料数据为synthetic data，峰值和误差不能作为真实7085Al材料、工艺、强化机制或最佳含量证据。实际可能不呈现典型过拟合曲线，必须保留真实结果。

## 当前学生任务

仅第1步：自行导入pandas，读取CSV到data08，预览前5行。先发代码检查再运行，不提供完整解法。尚无学生代码或输出；不要接着预填X/y、划分、拟合、曲线或任何数值性能。
