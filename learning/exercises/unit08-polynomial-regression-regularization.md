# 单元08练习与运行证据

## 当前状态

2026-10-27：学习中。一次与二次模型的训练/验证MAE均有学生实际输出并核实保存；一次训练MAE混用二次预测的首错保留，现已提示后修正。比较解释提示后完成，当前仅创建十二次转换器；尚无十二次运行、Ridge、RMSE、正式评估或单元完成。

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

## 2026-10-26：二次Pipeline训练真实执行与保存

### 聊天源码与修正层级

学生先正确回答二次展开预期shape为(30,2)。随后聊天提交PolynomialFeatures(degree=2, include_bias=False)、StandardScaler()、LinearRegression()各对象创建代码，静态检查正确；当时不冒充已经执行。

Pipeline首次列表填入PolynomialFeatures、StandardScaler、LinearRegression类名，顺序/结构正确但不是前面创建的对象。教师只解释类/实例/变量并提示后，学生将第二项改为poly08_2、scaler08_2、reg08_2；完整首答和修正版保留在错题第3项，不记首次独立正确。

之后学生原始训练语句：

```python
pipe08_2.fit(X08_train, y08_train)
```

静态检查通过后，教师让学生自行运行完整第4步格，没有代跑。

### 本次真实截图

codex-clipboard-4ce8a4a4-497e-4bd3-acdf-7ec90291800b.png显示输入/返回编号[18]，完整三实例、Pipeline正确顺序及仅训练集fit；返回Pipeline结构图，无可见异常。这是拟合成功的执行证据，不是UTS预测、误差或泛化验证。

### 磁盘只读核实

Notebook45810字节，mtime为2026-10-26 16:57:22 +08:00，SHA256为1E04EDE9FD5B9D0FD67E8D00863D70DBC6255E436101E54151EE9BC0EF062BF7。

已执行格编号5、8、10、18；[18]源码与截图一致，包含degree=2、include_bias=False的转换器、标准化器、线性回归实例，pipe08_2.fit只传X08_train/y08_train。输出为Pipeline表示，无error。Notebook尚无已填predict、评分或验证/测试fit，后续3个代码格仍未执行。原件及输出直接归档，不编辑学生Notebook。

核验并未重新执行代码，不证明干净内核顺序运行、当前内存完全同步、拟合质量、是否过拟合或真实材料有效性，也不据此声称Notebook之外从未使用测试数据。

### 当时下一步（后续执行见下节）

学生自行用pipe08_2对原始X08_train、X08_val调用predict，保存为y08_train_pred_2、y08_val_pred_2，并打印二者shape。先提交代码检查再运行；尚无该步骤代码、输出或指标，不预填数值。保持同一训练流程、不手工重复转换、不重新fit、测试封存。

## 2026-10-26：训练/验证预测与shape真实运行

### 首答与提示后修正

预测首次提交调用reg08_2，第二方法写成preedict；完整原答已保存于mistakes第4项。教师解释最后回归器与完整Pipeline的输入表示，提示改调用对象、方法拼写及补shape，没有代写完整预测代码。

学生下一版聊天原答：

```python
y08_train_pred_2 = pipe08_2.predict(X08_train)#变量名末尾的 _2 用于标记二次模型，方便之后与其他次数区分
y08_val_pred_2 = pipe08_2.predict(X08_val)
```

这两行静态正确，但尚缺shape打印；教师只让学生自行补打印后执行。

### 真实截图与结果

截图codex-clipboard-6d8d7de6-bfe0-408f-a6fb-c1476d8025f8.png保留完整对象创建、Pipeline、仅训练fit，随后两次完整流程predict及shape打印。输出：

```text
(30, 1)
(10, 1)
```

分别对应训练、验证预测，与二维目标相容；没有预测值、MAE/RMSE或泛化结论。截图执行编号裁切，不能由截图猜编号。

### 保存核实

教师独立只读Notebook11847字节、mtime2026-10-26 17:29:22 +08:00；SHA256 66B2EA02695F9DDAA81D5B4DAB815DB0CC78CDF76169CDC0DB596144E8A864F5。当前磁盘执行编号23、24、25、29；第4步[29]源码与截图一致，stream两行shape完整，无error、指标或测试预测/评分。

本次在同一格重新创建并训练了相同二次配置，两个predict共享当前这一个Pipeline；只拟合同一训练集，不是验证fit或泄漏，也不误记为完全没有再次fit。没有教师执行/编辑，执行编号不能证明干净重启验收。

提示后预测代码修正及实际shape检查完成，首次绕过Pipeline/拼写错误保留，不升级为首次独立通过。

当前只布置训练/验证MAE：使用sklearn.metrics.mean_absolute_error，结果变量mae08_train_2、mae08_val_2，打印并标明MPa；代码及数值尚未收到，不预填结果。不用测试、不计算RMSE或比较其他模型。

## 2026-10-27：两候选MAE、一次训练来源修正与保存

二次模型的mean_absolute_error配对正确，学生先用mae08_train命名，随后改为mae08_train_2，并自行补mae08_val_2。实际训练3.87910920966137、验证4.278087126458592。只缺次数后缀不是计算公式错误。

学生随后自行建立degree=1、include_bias=False的新转换器、标准化器、回归器和完整Pipeline，复用固定X08_train/y08_train及X08_val/y08_val。两个模型都仅训练集fit，完整Pipeline.predict接收原始输入，没有重复外部转换。预测shape均为(30,1)/(10,1)。

首次一次训练MAE误用了y08_train_pred_2；不是一次模型得到与二次相同训练成绩。原始代码、当时实际输出及解释保存在mistakes第5项。教师只提示右侧来源，未拟合参考模型或预填正确一次训练分数。

本次截图codex-clipboard-657c3dae-e33e-4ef0-8109-ba1aa7ad635c.png中修正版：

```python
mae08_train_1 = mean_absolute_error(y08_train,y08_train_pred_1)
print(mae08_train_1)
mae08_val_1 = mean_absolute_error(y08_val,y08_val_pred_1)
print(mae08_val_1)
```

截图注释未改变计算；上述仅摘录可执行语句。实际保存stream依次为：

```text
(30, 1)
(10, 1)
3.87910920966137
4.278087126458592
(30, 1)
(10, 1)
9.785853805627955
8.219585847395972
```

前四行二次，后四行一次。单位MPa为教师补充；y没有标准化，MAE与UTS同单位。全部synthetic data，非材料实验误差。

Notebook13421字节，mtime2026-10-27 10:03:54 +08:00，SHA256 CC9A0FCD31ACE090438F3B4D23C11DB9A6598F71B13F3AA23C1EAE48F86D4640；实际代码格计数2、3、4、5、6、7、8，模型/MAE同在第4步[5]，最后三个格仅任务注释，无输出。计数不证明干净重启或运行原因。教师仅只读核验，没有执行或编辑学生Notebook。

无已保存error对象；测试仅用于固定划分和shape打印，未做测试predict/评分。没有RMSE、高次/Ridge拟合或曲线。原Notebook字节直接归档，不改学生代码、注释、输出或其他单元。保存状态不冒充干净内核验收、真实材料效果或长期熟练。

两候选比较的首次原答已收到：“不能，需要验证三次四次等更高次的MAE”。判断正确但依据需澄清，原答及教师提示见quizzes/mistakes第6项；当时仅追问已有验证表现，后续真实答复见下节。本轮没有新代码、运行或数值，不改写/执行Notebook，不训练更高次。

## 2026-10-27：比较解释闭环及下一小步

学生提示后原答：“训练集和验证集都改善了”。本次判断正确，记提示后完成当前两候选比较检查，不覆盖首次“不能，需要验证三次四次等更高次的MAE”或升级为首次独立完整通过。教师补充：当前固定划分中两组MAE都下降，不能仅因训练误差更低就判二次过拟合；也不保证所有新数据表现或证明完全无过拟合。

当前任务：参照已有二次转换器，自行写一行创建poly08_12，degree=12且include_bias=False；先提交代码检查，不fit、transform、predict，不重划分、不覆盖现有对象。

尚未收到十二次代码，无十二次执行、shape或MAE，不预填结果、不代写或运行学生Notebook。本轮仅真实聊天答复，原Notebook及测试封存状态不变。
