# 单元08练习与运行证据

## 当前状态

2026-10-26：学习中。输入/目标shape与30/10/10划分正确，最新正确源码及输出已核实保存并归档。当前进入多项式特征转换前的shape检查，暂无转换代码、模型训练、预测、评分或正式单元评估。

## 独立资产（创建时状态；最新学生代码见文末）

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

## 第1步布置记录（历史）

仅第1步：自行导入pandas，读取CSV到data08，预览前5行。先发代码检查再运行，不提供完整解法。尚无学生代码或输出；不要接着预填X/y、划分、拟合、曲线或任何数值性能。

## 2026-10-26：第1步读取与预览已执行、已保存

学生提交截图codex-clipboard-2be65e9b-8cf8-436c-adcf-4d936a8c917a.png，输入/输出编号[5]。可见学生代码：

```python
import pandas as pd
data08 = pd.read_csv("data/unit08-materials-polynomial-synthetic.csv")
data08.head()
```

实际预览列名ZrB2_wt_pct、UTS_MPa，前5行分别为(0.000,467.88)、(0.041,472.25)、(0.082,477.30)、(0.122,483.61)、(0.163,481.89)。截图未见异常，和既有synthetic资产前5行一致。head()仅为预览，不能由5行预览推断全表只有5行。

教师随后只读本地Notebook：9739字节，保存时间2026-10-26 10:40:45；唯一已执行代码格execution_count=5，源码为上述三句及原任务注释，输出类型execute_result，无error对象。读取与预览已保存，不仅是口头报告；教师未代跑或编辑学生代码。编号5不证明重启内核后完整顺序运行，也不推断此前发生过哪些尝试。

当前只完成读取与预览。尚无学生实际查询全表shape、X/y选列、划分、转换、模型训练、预测或误差计算。

当时下一步为预测shape；后续真实答复、代码与划分证据见下节，原待答状态为历史。

## 2026-10-26：二维选列与两次划分

### 口头shape与实际选列

学生原答（原代码块标记为ini，内容按原样保存）：

```text
data08.shape = (50, 2)
X08.shape    = (50, 1)
y08.shape    = (50, 1)
```

三个预期判断正确。随后截图codex-clipboard-497e8b02-896a-40cf-b733-2eacfe4b236c.png可见X08选data08[["ZrB2_wt_pct"]]、y08选data08[["UTS_MPa"]]并分别打印三者shape，真实输出为(50,2)、(50,1)、(50,1)。代码及结果正确，没有新增错误。

### 第一次划分

截图codex-clipboard-12709b42-5dfa-4cf2-9939-f6898e9aa3cf.png可见从sklearn.model_selection导入train_test_split，同时传入X08/y08、test_size=0.4、random_state=42，按X08_train/X08_temp/y08_train/y08_temp接收。实际输出(30,1)、(20,1)、(30,1)、(20,1)。教师未执行或重算。

### 第二次划分截图

截图codex-clipboard-e95aa041-d2ad-4b3f-be3e-31559435cbf6.png在同格保留第一次划分，并增加：

```python
X08_val,X08_test,y08_val,y08_test = train_test_split(X08_temp,y08_temp,test_size = 0.5,random_state = 42)
print(X08_val.shape,X08_test.shape,y08_val.shape,y08_test.shape)
```

第二行实际输出四个(10,1)，首次分割输出也保留。截图源码/结果正确，两次固定划分合为30/10/10；本轮没有拟合转换器、模型或测试集评分。

### 保存核验差异（历史，已由下节闭环）

教师两次只读本地Notebook，均为10598字节、2026-10-26 11:02:45。读取格[5]、选列格[8]（三行shape正确）已保存。划分格[9]已存第二次split正确输入及接收名，但最后print仍是旧的X_val/X_test/y_val/y_test，没有08；保存输出只有第一次split的30/20一行，无error对象。

这是磁盘与最新截图之间的差异，不能反推正确截图仍有旧名错误，也不能因为未存error就认定旧无08名称有效/已有内存绑定；没有证据说明具体修改或执行顺序，不编造NameError。编号[9]不证明当前已存源码曾完整执行。

当前只请学生在已有正确截图的Notebook按Ctrl+S并回复已保存，再只读核对。不要改写/代跑学生Notebook，也不把旧存盘版本作为最新成果上传。尚未收到保存回报或多项式转换代码。

## 2026-10-26：最新划分保存已核实

学生原答：

> 已保存

教师随后只读本地Notebook：10650字节，LastWriteTime为2026-10-26 11:48:51，SHA256为E97340982E13A56D42BB3DFE5129C1E8A7134F94F40921640344541EE2E0FD0E。

读取格[5]、选列格[8]保留，最新划分格[10]的两次split与两条print均使用X08/y08本单元变量，源码和正确截图一致。实际保存输出：

```text
(30, 1) (20, 1) (30, 1) (20, 1)
(10, 1) (10, 1) (10, 1) (10, 1)
```

无error输出，前次旧print与缺第二行结果的保存差异现已闭环；保留历史，不倒改为此前一直保存正确。此为学生保存报告加磁盘源码/输出证据，教师没有运行、编辑或补造Notebook输出。编号5/8/10不等同干净内核顺序运行，最终验收仍留单元末。

当前只进入二次多项式展开shape：X08_train为(30,1)，degree=2、include_bias=False时需要学生先判断展开形状。尚未收到答案或多项式代码，不预填通过。后续正式训练中的转换和标准化必须同置Pipeline，不在验证/测试上fit。
