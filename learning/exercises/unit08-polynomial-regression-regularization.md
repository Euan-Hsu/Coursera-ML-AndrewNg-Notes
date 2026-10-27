# 单元08练习与运行证据

## 当前状态

2026-10-27：学习中。英文标题/图例修复已核实保存，当前绘图输出不再含缺字警告，三模型误差未变。当前只观察十二次与二次曲线右端形状；尚无学生本次读图答复、Ridge、RMSE、正式评估或单元完成，测试保留。

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

当时下一小步是自行创建十二次转换器，后续真实代码与执行见下节。

该任务布置当时尚未收到十二次代码或执行，不预填结果；后续真实进展见下，不倒改当时证据。

## 2026-10-27：十二次Pipeline静态检查及真实训练保存

学生首先提交原文：

```python
poly08_12 = PolynomialFeatures(degree = 12, include_bias = False)#创建十二次模型
```

构造代码正确。教师只将注释用语澄清为创建多项式特征转换器，尚非完整回归模型；没有改学生代码或代创建模型。之后学生自行提交：

```python
scaler08_12 = StandardScaler()#创建标准化器

reg08_12 = LinearRegression()#创建回归器

pipe08_12 = Pipeline([
    ("poly", poly08_12),
    ("scale", scaler08_12),
    ("reg", reg08_12),
])
pipe08_12.fit(X08_train, y08_train)
```

新对象、顺序、仅固定训练集fit均正确。学生主动同时写fit语句，教师检查后要求自行运行并提供输出；额外写出正确训练语句不是模型实现错误。聊天文本当时仅静态证据，后续截图及保存才闭环。

截图codex-clipboard-21d3b2d7-9deb-42c2-a6f5-1009ad7c0574.png显示第4步[5]返回Pipeline，无可见异常。磁盘Notebook48358字节、mtime2026-10-27 10:47:48 +08:00，SHA256 13A4AFCA633FE6F3897906DBCCB75B0319B141A74BA68A78E17212AD28796EF4。实际代码格计数2/3/4/5/6/7/8，第4步同格保留二次/一次代码、八行stream和新增十二次训练，最后返回十二次Pipeline表示；后三格仅注释。

原八行shape/MAE仍属于二次与一次：二次3.87910920966137/4.278087126458592，一次9.785853805627955/8.219585847395972（训练/验证，MPa由教师补充）。没有十二次预测、shape或误差，不能把旧输出当十二次成绩。源码仅训练集fit；测试仅既有划分/shape，未见测试预测或评分。所有材料数据synthetic data，不作真实材料性能、机制或最佳含量证据。教师仅只读核验，未编辑/执行Notebook或拟合参考模型，不据计数推断重启、具体操作顺序或Notebook之外状态。

原件直接归档，原一次/二次代码与结果没有被覆盖；不升级为从零独立完成所有建模任务或单元末验收。

当时仅布置十二次预测并等待代码；后续真实截图及保存差异见下节，保留当时证据层级。

## 2026-10-27：十二次预测、MAE真实截图与保存差异

正确截图a98bd5ae-91c8-4c28-a261-c866f80ea56d.png先显示完整pipe08_12.predict接收原始训练/验证输入，结果名均_12，新增预测shape(30,1)/(10,1)。最新截图def6904e-b986-4017-a8a8-0915b97ff123.png中mean_absolute_error分别配对y08_train/y08_train_pred_12和y08_val/y08_val_pred_12，打印mae08_train_12、mae08_val_12，实际末两行2.6663586323270865、4.190316693931817。截图执行编号未见，不从别处推定。全部synthetic data；MPa由教师根据未缩放UTS目标补充，原输出为裸数。

截图中已正确实现的评分语句（仅摘录可执行语句）：

```python
mae08_train_12 = mean_absolute_error(y08_train,y08_train_pred_12)
print(mae08_train_12)
mae08_val_12 = mean_absolute_error(y08_val,y08_val_pred_12)
print(mae08_val_12)
```

截图stream保留前两个模型及十二次预测shape，实际末两行是十二次MAE，不与旧数值混记。

| 模型次数 | 训练MAE（MPa） | 验证MAE（MPa） |
| --- | ---: | ---: |
| 1 | 9.785853805627955 | 8.219585847395972 |
| 2 | 3.87910920966137 | 4.278087126458592 |
| 12 | 2.6663586323270865 | 4.190316693931817 |

磁盘只读核验仍为14550字节、mtime2026-10-27 10:57:49 +08:00，SHA256 3D84E87CEC66C97244DBF933CAAC847D32222DB7690E2E49C7B57B4DC37F77A4，代码格计数2/3/4/9/6/7/8。第4步[9]已保存正确十二次预测/shape，但末尾评分源码仍是mae08_train_12 = mean_absolute_error(y08_train,y08_train_pred_1)，随后print(mae08_train_1)、一次验证评分及print(mae08_val_1)。stream只有旧一次/二次MAE及新增十二次两行shape，缺十二次MAE。保存源码与该stream不能证明修正版已执行保存。此为当前正确截图与磁盘的保存差异，不判截图代码仍错，也不编造异常、内存状态、重启或具体编辑/执行顺序。

当前仅要求保存，不教师代改/代跑、不拟合参考模型、不上传落后Notebook冒充最新成果。源码中fit只训练、预测/评分仅train/val，测试只既有划分和shape；无测试预测评分、Ridge或RMSE。单元末干净重启验收未做，不声称Notebook外绝无测试使用。

当时保存任务：学生在当前显示正确十二次MAE的Notebook按Ctrl+S并回复“已保存”；无需修改截图中正确代码或重新运行。随后教师只读核对源码与输出，保存闭环后再解释二次与十二次比较。

## 2026-10-27：十二次预测/MAE保存差异闭环

学生原答：“已保存”。教师随后只读Notebook14615字节、mtime2026-10-27 11:01:53 +08:00，SHA256 F7022303F351FC207BC70F4F2C938234C05B3E57B15A706BA64BD883B6294236。代码格计数2/3/4/10/6/7/8，第4步[10]的十二次predict、MAE目标/预测配对和print变量均正确用对应_12，保存stream末四行为(30,1)、(10,1)、2.6663586323270865、4.190316693931817。与先前正确截图一致，无error输出；前两个模型指标不变。后三格仍仅注释。

本次已保存评分语句：

```python
mae08_train_12 = mean_absolute_error(y08_train,y08_train_pred_12)
print(mae08_train_12)
mae08_val_12 = mean_absolute_error(y08_val,y08_val_pred_12)
print(mae08_val_12)
```

末四行真实stream：

```text
(30, 1)
(10, 1)
2.6663586323270865
4.190316693931817
```

这是学生保存报告加磁盘源码/输出核验，不是教师代跑、修改或补造输出，不据计数推断干净重启、点击顺序或Notebook外状态。实质拟合只训练，预测/评分仅训练与验证；无Ridge实现、RMSE或测试预测评分。全部synthetic data，MAE单位MPa由教师补充，不能作真实材料性能/最佳含量证据。

原Notebook字节归档，不修改学生代码或已有一次/二次结果。旧磁盘差异保留为历史，不将本次正确截图再记新错误。

当时仅布置比较问题，后续答复和绘图见下；不倒改当时未收状态。

## 2026-10-27：比较解释、绘图网格与实际三曲线

比较题学生原答：“没有出现明显过拟合，虽然训练MAE下降，但是验证MAE同步下降”。本次教学检查核心正确，学生同时引用了两组误差变化；教师将措辞限定为当前固定划分未显示明显过拟合证据，不保证完全无过拟合或稳定最优。不能回写此前第6项首次依据偏离，也不冒充正式单元末评估。

### 网格构造：保留两版原稿

先答200点单原始特征的shape为(200,1)，判断正确，非当时运行。

第一版原始提交：

```python
import numpy as np
import matplotlib.pyplot as plt
np.linspace(0, 2, 200) #生成200个等间隔点,np.linspace(起点, 终点, 点数)
pd.DataFrame({"ZrB2_wt_pct": 一维数值数组}) #保存为单列DataFrame,pd.DataFrame({"列名": 一维数值数组})
print(ZrB2_wt_pct.shape);一维数值数组不知道填什么
```

末尾中文是学生提问，不能将整段当作已运行程序或编造语法/名称异常。核心缺口是先保存linspace返回值，再传给DataFrame；起止范围需来自训练列min/max。教师只提示第一步，没有直接补完整网格程序。

下一版原始提交：

```python
import numpy as np
import matplotlib.pyplot as plt
X08_grid = np.linspace(X08_train["ZrB2_wt_pct"].min(), X08_train["ZrB2_wt_pct"].max(), 200) #生成200个等间隔点,np.linspace(起点, 终点, 点数)
pd.DataFrame({"ZrB2_wt_pct": X08_grid}) #保存为单列DataFrame,pd.DataFrame({"列名": 一维数值数组})
print(X08_grid.shape)
```

训练范围与数组保存已正确，X08_grid大小写相对提示不同但一致使用合法，不判错。教师指出主动附带的DataFrame/print尚未达到二维输入目标：前者结果未保存、后者只查看数组，先让只改后两行。学生随后实际截图[23]已用X08_plot = pd.DataFrame({"ZrB2_wt_pct": X08_grid})，保留两个shape打印，输出(200,)/(200,1)；额外打印数组shape无误。

### 三模型网格预测

学生原始三行正确：

```python
y08_plot_pred_1 = pipe08_1.predict(X08_plot)
y08_plot_pred_2 = pipe08_2.predict(X08_plot)
y08_plot_pred_12 = pipe08_12.predict(X08_plot)
```

随后自行补shape打印，截图64114d79-1b6d-48f1-b2cb-6985f0f2426e.png实际三组均(200,1)，没有网格fit或手动再次变换。网格只是绘图位置，无新增真实目标，不用它算“实验误差”。

### 散点、曲线与标题

scatter首稿：

```python
plt.scatter(X08_train,y08_train,label = "训练集和验证集的样本散点图")#散点函数
```

只传训练组，label不会自动加入验证数据；当前单列DataFrame可用于scatter，不虚构shape错误。教师提示后学生改为：

```python
plt.scatter(X08_train,y08_train,label = "训练集")#训练集散点函数
plt.scatter(X08_val,y08_val,label = "验证集")#验证集散点函数
```

两组输入/目标配对和标签正确，记提示后修正。之后学生自行写三条plt.plot，统一横坐标X08_plot["ZrB2_wt_pct"]、各自y08_plot_pred_1/_2/_12与模型标签，均正确。图前plt.figure、轴含量wt.%和UTS MPa、legend/show均正确。最初标题“训练集和验证集的样本散点”缺synthetic data，教师仅让改标题；本次截图已为含synthetic data的多项式拟合对比标题。

### 本次真实输出与字体警告

最新磁盘Notebook67988字节，mtime2026-10-27 13:10:56 +08:00，SHA256 3A7A6F8D9B155B9FDC99DF4543E4B1B6FB87D885BAF16595281F992D262A1050；代码格计数13/14/15/16/30/null/null，绘图格[30]已存stdout、stderr和display_data，PNG571×455、35052字节。stdout为(200,)、(200,1)、(200,1) (200,1) (200,1)，无error输出；stderr为UserWarning: Glyph ... missing from font(s) DejaVu Sans。

网格仅由训练集ZrB2_wt_pct的min/max生成，单列同名DataFrame交给三个已训练完整Pipeline预测；绘图格没有fit，散点只含训练/验证，三条线共享按含量递增的同一网格。既有三个模型MAE与此前完全一致，无Ridge/RMSE或测试预测评分。全部synthetic data；图标题最终已含synthetic data，轴含量wt.%、UTS MPa。教师仅只读核验，没有执行/编辑Notebook、重拟合参考模型或制造输出，不从计数推断重启/点击顺序或Notebook外状态。

图已生成，中文标题/图例部分显示方框；这是字体缺少中文字形的显示警告，不是回归训练异常。保留含警告的原始Notebook，不自行清理或改输出。学生还没有解释图形，不代填读图结论。

当时字体任务：仅将五个label字符串改为Training、Validation、Degree 1、Degree 2、Degree 12，标题改为Polynomial regression comparison - synthetic data；只修改文字，重新运行当前绘图单元格并发实际图像，不改模型/数据、不重跑训练，不调用测试。该任务布置时尚无修改后图像或警告消除证据，后续闭环见下。

## 2026-10-27：英文标注修复已执行并保存

学生原答：“已修改保存”。教师只读核实Notebook72888字节、mtime2026-10-27 13:18:58 +08:00，SHA256 D8DA35CE6C49CB383C461125C458D2386861D955720F54D792619A45FB537808；代码格计数13/14/15/16/31/null/null。绘图格[31]源码label为Training、Validation、Degree1、Degree2、Degree12，标题Polynomial regression comparison - synthetic data；实际查看已存571×455 PNG（42155字节），英文标题/图例及wt.%/MPa单位显示正常。最新输出只有stdout和display_data，无stderr/glyph警告或error。stdout仍为(200,)、(200,1)、(200,1) (200,1) (200,1)。

三个模型MAE与前一检查点完全一致，网格仍只由训练范围生成、仅用于完整Pipeline.predict，绘图单元格无fit、验证fit或测试使用；无Ridge/RMSE/测试预测评分。教师未执行/编辑模型或绘图，也不据计数推断干净重启/具体操作顺序。先前中文警告原件仍保留在Git提交edf92b58b3b23ec78df7160d2265d1c19aa1ace8，不将当前输出没有警告误写成此前从未出现。全部synthetic data，不作真实材料最佳含量/强化机制或性能证据。

本次只替换标注字符串，实际绘图/网格预测由学生执行。Notebook原字节归档，不教师改图、改输出或删除过去错误。字体任务完成，不等同正式单元末绘图评估或读图能力通过。

当前只让学生观察二次/十二次右端形状，具体题面见quizzes；未收到读图答案，不重新建模或开启测试。
