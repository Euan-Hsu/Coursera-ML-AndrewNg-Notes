# 单元08练习与运行证据

## 当前状态

2026-10-29：学习中。P2代码已提示后修正，沿用P1已训练pipe08_check预测新synthetic输入，输入与预测shape均(2,1)，实际输出已保存；MAE判断首次正确。P2整题提示后完成，正式已答7/8、通过7/8（无内容提示完成4、提示后完成3）；当前最后第8题T1材料迁移待首次作答，最终独立测试仍待。

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

当时只布置右端形状观察；后续真实答复与评分范围检查见下，不重新建模或开启测试。

## 2026-10-27：曲线形状描述与评分范围检查

学生对橙色平缓下降、绿色先降后升的描述正确，逐字答复见quizzes。教师未把图形现象冒充真实材料机制或仅凭形状改写验证分数。

新问题是无验证样本区间是否一定令验证MAE明显增加；学生首答认为会显著增加，原文/提示保留在mistakes第9项。

教师指出不能由缺少验证样本直接推断MAE必然显著增大，并提示回看已执行的mean_absolute_error(y08_val,y08_val_pred_12)：两个参数各对应10个验证样本，而非200个绘图网格预测。仅让学生检查实际传入哪些点，不代写完整因果解释或预填修正答案。

当时只问评分范围最小问题；后续真实答复与新任务见下，原未收状态保留为历史。

本轮仅聊天概念检查，无新代码、运行、模型指标或Notebook修改，已有英文图及MAE不变；全为synthetic data。既不能把缺覆盖说成MAE必然增大，也不把它反写成MAE必然低/必然不变或证明完全无过拟合。保持首次观察正确、随后因果推断错误两层，不否认此前正确比较，也不推断学生未表达的思路。

## 2026-10-27：评分范围理解收口与Ridge准备任务

学生依次真实答“不会”和“验证MAE较低，可能是因为有些异常预测没有反映在验证MAE中”。最小问题“该谷底绘图点是否参与10个验证样本的平均”在提示后答“不会”，判断正确。随后自主复述方向正确，但未明确说出验证覆盖位置与评分样本的机制；记提示后基本理解、表述由教师补精确，不覆盖首次错误或标首次独立完整通过，也不继续反复追问阻止后续学习。

教师澄清：这里指没有验证样本覆盖的位置未被评分；MAE计入所有实际提供的验证样本，不会自动剔除预测很差的验证点。缺覆盖不意味着MAE一定很低/不变或证明完全无过拟合，模拟深谷不能当材料实验事实。

当时仅布置Ridge导入/对象创建，后续真实源码检查见下；不倒改当时没有执行证据的状态。

本轮仅聊天解释与下一小步布置，没有运行/修改Notebook、创建教师模型或新增数值。已有三模型MAE与英文图不变，全部synthetic data；测试不预测/评分，不保证Ridge验证误差必降。后续仍固定十二次、Pipeline内训练拟合转换/标准化，alpha候选按原0.01/0.1/1/10/100/1000，不换seed挑结果。

## 2026-10-27：Ridge对象创建与Pipeline首稿静态检查

先前学生单独提交from sklearn.linear_model import Ridge及ridge08_12_a1 = Ridge(alpha = 1)，写法正确。教师仅布置独立转换器/标准化器和新Pipeline，不代写完整代码。

本次完整原稿：

```python
from sklearn.linear_model import Ridge
ridge08_12_a1 = Ridge(alpha = 1)

poly08_12_a1 = PolynomialFeatures(degree = 12, include_bias = False)#新的多项式转换器

scaler08_12_a1 = StandardScaler()#新的标准化器

reg08_12_a1 = LinearRegression()#新的回归器

pipe08_12_a1 = Pipeline([
    ("poly", poly08_1),
    ("scale", scaler08_1),
    ("reg", reg08_1),
])#按“多项式展开 → 标准化 → 回归”组装
```

Ridge导入和alpha=1对象、degree12/include_bias=False新转换器及新标准化器的写法均正确；额外LinearRegression仅为这段尚未使用的对象，不等于Ridge。Pipeline结构/顺序正确，但第二项仍分别引用旧poly08_1、scaler08_1、reg08_1，实际接入旧一次普通线性候选，而非当前十二次Ridge。外层变量名_12_a1不会自动替换内部对象。

当时仅提示对象引用修正；后续两版及执行证据见下，不倒改当时证据。

本次仅聊天源码静态检查，无执行/报错/保存/训练证据，不编造TypeError、实际拟合、原模型已覆盖或数据泄漏，也不预填Ridge预测/MAE。教师未修改/执行Notebook或创建参考模型；既有三个模型和synthetic data英文图保持。固定训练/验证/测试、预定alpha及仅训练fit规则不变，测试继续封存。

## 2026-10-27：Ridge第二版、关键单行提示与正确训练

第二次Pipeline原文摘录（上方创建仍保留了普通LinearRegression对象）：

```python
pipe08_12_a1 = Pipeline([
    ("poly", poly08_12_a1),
    ("scale", scaler08_12_a1),
    ("reg", reg08_12_a1),
])#按“多项式展开 → 标准化 → 回归”组装
```

第二版已把poly/scaler改成本次新对象，前两步正确，但末步仍是由LinearRegression()创建的reg08_12_a1，Ridge未接入。教师先认可前两处，再直接给出最后一项("reg", ridge08_12_a1)，说明步骤名称不决定算法、本候选无需额外普通LinearRegression。第三版学生接入Ridge后静态正确。此为提示并给出关键单行后的修正，不回写成整条Pipeline首次独立正确；首版与第二版均保留。

第三次学生Pipeline原文：

```python
pipe08_12_a1 = Pipeline([
    ("poly", poly08_12_a1),
    ("scale", scaler08_12_a1),
    ("reg", ridge08_12_a1),
])#按“多项式展开 → 标准化 → 回归”组装
```

检查正确后教师只布置仅训练集fit语句，学生截图已自行补：

```python
pipe08_12_a1.fit(X08_train, y08_train)
```

截图45fb6653-f9c9-4db4-9410-e6fc0e4718a1.png显示正确独立工具和pipe08_12_a1.fit(X08_train,y08_train)返回含Ridge的Pipeline；截图编号左侧裁切，不根据截图猜编号。磁盘Notebook109256字节、mtime2026-10-27 15:18:06 +08:00、SHA256 250ED6027A504DBFBCBA4C50E7BF01FE5B68961A2659E7398421D302E05E0F14；代码计数13/14/15/16/31/32/null，Ridge格实际[32]。保存源码degree12/include_biasFalse、新scaler和Ridge(alpha1)、三个引用均正确，仅fit原固定训练组；execute_result的plain text明确为PolynomialFeatures→StandardScaler→Ridge(alpha=1)，无error。

尚无Ridge predict、shape、MAE/RMSE、其他alpha或Ridge曲线；测试格仍未填，未见验证fit、网格fit或测试预测评分。原三个OLS候选及英文图仍保留。教师仅只读核验、未执行/编辑Notebook或补算指标，不据计数推断重启/点击顺序/Notebook外状态，不由fit成功认定正则化更好。数据全为synthetic data，固定划分与预定alpha候选不变，测试留到最终固定方案后一次评价。

原Notebook直接归档，不修改学生代码/输出或旧模型。当时只布置Ridge预测，最新代码/shape见下，保留当时未收状态。

## 2026-10-27：Ridge实际预测shape与实现核验

本次学生可执行语句摘录：

```python
y08_train_pred_12_a1 = pipe08_12_a1.predict(X08_train)
y08_val_pred_12_a1 = pipe08_12_a1.predict(X08_val)
print(y08_train_pred_12_a1.shape, y08_val_pred_12_a1.shape)
```

截图654b00de-a62e-4231-9c0c-0d213d531a36.png中完整pipe08_12_a1.predict接收原始训练/验证输入，变量后缀均_12_a1，print实际输出(30,) (10,)。磁盘Notebook73794字节、mtime2026-10-27 15:40:07 +08:00、SHA256 F12CB0184E9FBBAE84A37309A882BF0C85C995E00AC3B00C536479F6E55C1BC5；Ridge格实际[35]源码/保存stdout与截图一致，无error。y08源码仍双括号选UTS_MPa的二维单列，已有训练/验证目标shape仍(30,1)/(10,1)，fit仍使用相应训练组。

实际保存stdout：

```text
(30,) (10,)
```

教师只读当前.venv的scikit-learn1.7.2源码：linear_model/_ridge.py第809–810行在n_targets==1时展平系数，linear_model/_base.py第279–280行据一维系数返回一维预测。故本次(样本数,)是实现允许的单目标表示，不是少预测一列、特征展开丢失或学生把目标改成一维。不能把之前沿用的二维预测约定误说成所有回归器原生输出的硬要求。

另外只读metrics/_regression.py第120–124行可见，评分函数会把单目标的一维真实值/预测转为列。因此本次显式reshape不是mean_absolute_error的硬性输入要求，只是本单元统一布局和后续手工数组运算的约定；不能说原生一维预测无法评分。

当时只布置形状整理，后续学生求助及教师代码帮助后执行见下，不倒改当时证据。

此处整理是布局统一而非修复模型。尚无Ridge MAE/RMSE、其他alpha或Ridge曲线，测试仍未预测评分。教师只读文件/依赖源码，未执行模型、修改Notebook或补造输出；不据计数推断重启/操作顺序或Notebook外状态。全部synthetic data，既有三个OLS模型和英文图保留。

## 2026-10-27：reshape帮助后完成与Ridge MAE真实运行

在原生一维预测已正确的前提下，教师布置统一二维列的reshape练习，学生原答“不会”；这是求助而非已提交错误程序。教师随后给出两行reshape赋值和shape打印的完整写法，学生截图成功运行。该布局整理记为教师直接代码帮助后执行，不回写独立实现，也不说原生Ridge一维输出错误或reshape为MAE硬要求。

教师在学生“不会”后给出的这一步完整代码为：

```python
y08_train_pred_12_a1 = y08_train_pred_12_a1.reshape(-1, 1)
y08_val_pred_12_a1 = y08_val_pred_12_a1.reshape(-1, 1)

print(y08_train_pred_12_a1.shape, y08_val_pred_12_a1.shape)
```

后续学生成功截图中同格也保留了两行predict，输出(30,1)/(10,1)，不误记为完全没有再次predict。随后学生自己提交正确MAE语句，摘录可执行部分：

```python
mae08_train_12_a1 = mean_absolute_error(y08_train,y08_train_pred_12_a1)
print(mae08_train_12_a1)
mae08_val_12_a1 = mean_absolute_error(y08_val,y08_val_pred_12_a1)
print(mae08_val_12_a1)
```

截图3004f802-3f73-446d-82e4-084934544f01.png已显示两个预测reshape后(30,1)/(10,1)；本次截图87297e4f-6444-4568-8ccb-229c5f0b28aa.png的MAE目标/预测/打印变量均正确配对_12_a1，实际训练5.110107645592669、验证4.781618618396107。磁盘Notebook74478字节、mtime2026-10-27 16:20:09 +08:00、SHA256 C59B10EE227E627B224680A487C41F126BB885BA6D5E5C95A8A2F7598B01CFDD；代码计数13/14/15/16/31/38/null，Ridge格[38]同源码/stream，无error。

保存stdout：

```text
(30, 1) (10, 1)
5.110107645592669
4.781618618396107
```

| 十二次候选 | 训练MAE（MPa） | 验证MAE（MPa） |
| --- | ---: | ---: |
| LinearRegression，无正则化 | 2.6663586323270865 | 4.190316693931817 |
| Ridge，alpha=1 | 5.110107645592669 | 4.781618618396107 |

实际Ridge格仍同格创建新工具、仅fit固定训练集，再predict原始train/val、reshape并计算MAE；不能误记成只执行reshape或没有同格fit/predict。相同训练组重新拟合同配置不判泄漏，既有三个OLS指标及英文图不变。尚无其他alpha、Ridge网格预测/曲线、RMSE或测试预测评分。教师只读核验、未执行/改写Notebook或补造模型输出，不据计数推断重启/操作顺序/Notebook外状态。数据均synthetic data，y未缩放，MAE单位MPa由教师补充，原输出为裸数。

原Notebook与输出直接归档，不教师重算、删去“不符合预期”的结果或改seed。当时只布置验证表现比较，最新首次答复与提示见下，保留当时未收状态。

## 2026-10-27：Ridge MAE已算对，比较方向待修正

学生对比较题的完整原答：

> 改善了，虽然整体MAE提高，但是验证MAE比训练MAE有所下降，排除了过拟合

学生指出Ridge验证4.7816小于自身训练5.1101，这个数值大小关系正确；但本题是比较同一验证集上两个候选，不能用模型内train/val的不同组误差代替跨模型验证比较。“排除了过拟合”也无充分依据，验证低于训练不能作此绝对诊断。原话“整体MAE”未给定义或计算，不新增/虚构这一聚合指标。

教师认可其观察到的模型内大小关系，但指出比较方向应为两个模型的同一验证列，重列未正则十二次4.1903、Ridge alpha1为4.7816 MPa，提醒MAE越小越好；说明train/val来自不同样本，验证低于训练不能排除过拟合。先只让学生重新判断本次候选，不重开整套理论或给其他alpha结果。

当时只问同一验证列比较；后续真实答复见下，保留当时未收状态。不要为调整认知解释而重新训练或篡改已有数值。

本轮仅聊天评价，无新代码/执行/指标或Notebook修改，所有数值来自既有synthetic data真实运行。既有正确MAE配对、形状整理帮助及第9/10项层级不改；不能因本次解释错误否认计算源码正确，也不由单个alpha宣称全部正则无效/必然欠拟合。固定划分、候选计划和最后一次测试评价规则不变，不换seed/数据或提前看测试。

## 2026-10-27：Ridge比较提示后正确、网格预测及首版对照图

学生比较题原答：

> 没有改善，MAE比未正则化后增大了

学生提示后重新按同一验证列比较，方向与结论正确；教师将措辞明确为验证MAE从4.1903升至4.7816 MPa。记第11项提示后正确，不覆盖首次“改善了/排除了过拟合”，也不由alpha1泛化为所有正则化无效。

教师随后只布置同一网格预测、按已讲reshape方法整理二维，学生可执行语句摘录：

```python
y08_plot_pred_12_a1 = pipe08_12_a1.predict(X08_plot)
y08_plot_pred_12_a1 = y08_plot_pred_12_a1.reshape(-1, 1)
print(y08_plot_pred_12_a1.shape)
```

代码正确，实际(200,1)；这是依据已讲方法的迁移，不能回写最初reshape曾无需帮助，也非正式单元末独立Python评估。

教师布置在Ridge预测之后新建对照图（不把后定义变量提前塞入旧图），保留两组样本散点及两条曲线。学生首版截图可执行绘图语句摘录（省略注释）：

```python
plt.figure()
plt.plot(X08_plot["ZrB2_wt_pct"], y08_plot_pred_12, label="Degree12 OLS")
plt.plot(X08_plot["ZrB2_wt_pct"], y08_plot_pred_12_a1, label="Degree12 Ridge alpha=1")

plt.xlabel("ZrB2 content (wt.%)")
plt.ylabel("UTS (MPa)")
plt.title("Ridge comparison - synthetic data")
plt.legend()
plt.show()
```

截图43a56f80-a7a5-4ad7-b7ac-a8a056abaca1.png正确用pipe08_12_a1预测同一X08_plot、保存返回值并reshape，实际(200,1)。新截图17e1a309-4962-43f4-9996-3c0f8e464240.png的独立对照图有正确OLS12/Ridge alpha1两条曲线、英文图例、含synthetic data标题及wt.%/MPa轴标，但遗漏任务要求的训练/验证散点。

磁盘Notebook122821字节、mtime2026-10-27 17:19:59 +08:00、SHA256 7B8CF9998000404892A5FDD707353DAA63DD533C30961B3FDCF90CB9A1B03C23；代码计数13/14/15/16/31/45/null，Ridge格[45]源码/已存图像一致。stdout为(30,1) (10,1)、5.110107645592669、4.781618618396107、(200,1)，新PNG571×455/35564字节，无stderr或error。原三模型图[31]及其571×455/42155字节PNG仍保留。

程序运行并生成两曲线图，曲线/图例/单位/title均正确，唯一待补是任务要求的两组散点；缺散点不等于模型或曲线计算错误。原始图像和代码直接归档，前面三模型图保留。

当时仅布置补散点，最新真实修正和图像见下，保留当时未收状态。

当前Ridge格仍同格创建工具、仅原训练集fit，再预测train/val、reshape/MAE和网格predict/reshape，最后绘新图；网格没有fit，不误记成整格未重新训练。原数据、固定划分和三个OLS结果保持；无其他alpha、RMSE或测试预测/评分。教师仅只读核验/归档，未执行/编辑模型或图像，不凭计数推断重启/操作顺序。数据全部synthetic data，平滑曲线不自动代表验证误差改善，也不当真实材料机制/最佳含量。

## 2026-10-27：Ridge对照图散点补齐并保存

学生按提示在新figure后、两条plot前自行添加的可执行语句摘录：

```python
plt.scatter(X08_train,y08_train,label = "Training")
plt.scatter(X08_val,y08_val,label = "Validation")
```

配对/顺序正确，现新图由训练/验证点、OLS12/Ridge1曲线、轴单位和来源标题组成。记提示后补全，不覆盖缺散点首稿或当成整图首次独立正确。

截图1307bbce-3c4a-440a-a1c1-f655d4b300a9.png已在新figure后、两条plot前补正确Training/Validation散点。磁盘Notebook128787字节、mtime2026-10-27 17:56:17 +08:00、SHA256 E39F2C4F86D6F9C28EC27339782E2352837F879800ED10C851982B2063065F42；代码计数13/14/15/16/31/46/null，Ridge格[46]源码/内嵌图一致。新PNG571×455/39905字节，实际样本点、两曲线英文图例、synthetic data标题及wt.%/MPa轴均正常，无stderr/error。

stdout仍为(30,1) (10,1)、5.110107645592669、4.781618618396107、(200,1)，原三个OLS MAE未变，原三模型图[31]的PNG571×455/42155字节仍保留。同格仍只用训练组fit，再train/val预测/评分、网格预测及绘图，网格不fit；无其他alpha、RMSE或测试预测评分，测试格未填写。教师仅只读核验/归档，没有运行/修改模型或图，不据计数推断重启/操作顺序。全部synthetic data，不能由视觉平滑直接改写验证误差或推出真实材料机制/最佳含量。

无散点首图仍保存在此前Git提交d173e18aa77b9f6600e1c7a83ae5acc3fdbed92b，教师没有改图补点或清理历史。当时仅布置曲线观察，最新答复与候选首稿见下，不倒改当时未收状态。

## 2026-10-27：alpha0.1候选首次完整源码（尚无运行证据）

学生先答Ridge后“正则化后，曲线平缓下降”正确，随后答alpha1→0.1约束“会减弱”正确。教师只布置新候选的对象/Pipeline，约定a0p1表示alpha0.1；学生主动扩展到fit/predict/MAE/绘图，完整原稿如下，不当作已执行程序：

```python
#创建alpha = 0.1的新候选
ridge08_12_a0p1 = Ridge(alpha = 0.1)#新的回归器

poly08_12_a0p1 = PolynomialFeatures(degree = 12, include_bias = False)#新的多项式转换器

scaler08_12_a0p1 = StandardScaler()#新的标准化器

pipe08_12_a0p1 = Pipeline([
    ("poly", poly08_12_a0p1),
    ("scale", scaler08_12_a0p1),
    ("reg", ridge08_12_a0p1),
])#按“多项式展开 → 标准化 → 回归”组装
pipe08_12_a0p1.fit(X08_train, y08_train)

#Ridge预测
y08_train_pred_12_a0p1 = pipe08_12_a0p1.predict(X08_train)
y08_val_pred_12_a0p1 = pipe08_12_a0p1.predict(X08_val)
#为保持本单元的二维列约定，下一步只整理已有预测的形状,.reshape(-1, 1),-1 表示自动确定行数，1 表示一列
y08_train_pred_12_a0p1 = y08_train_pred_12_a0p1.reshape(-1, 1)
y08_val_pred_12_a0p1 = y08_val_pred_12_a0p1.reshape(-1, 1)
print(y08_train_pred_12_a0p1.shape, y08_val_pred_12_a0p1.shape)

#计算MAE
mae08_train_12_a0p1 = mean_absolute_error(y08_train,y08_train_pred_12_a0p1)#计算训练集MAE:结果变量 = 函数名称(真实值, 预测值)
print(mae08_train_12_a0p1)
mae08_val_12_a0p1 = mean_absolute_error(y08_val,y08_val_pred_12_a0p1)#计算验证集MAE
print(mae08_val_12_a0p1)

#观察Ridge曲线如何变化
y08_plot_pred_12_a0p1 = pipe08_12_a0p1.predict(X08_plot)
y08_plot_pred_12_a0p1 = y08_plot_pred_12_a0p1.reshape(-1, 1)
print(y08_plot_pred_12_a0p1.shape)

plt.figure() #创建新图
plt.scatter(X08_train,y08_train,label = "Training")#训练集散点函数
plt.scatter(X08_val,y08_val,label = "Validation")#验证集散点函数

plt.plot(X08_plot["ZrB2_wt_pct"], y08_plot_pred_12, label="Degree12 OLS") 
plt.plot(X08_plot["ZrB2_wt_pct"], y08_plot_pred_12_a1, label="Degree12 Ridge alpha=0.1") 

plt.xlabel("ZrB2 content (wt.%)") #标注横轴
plt.ylabel("UTS (MPa)") #标注纵轴
plt.title("Ridge comparison - synthetic data") #添加标题
plt.legend() #显示图例
plt.show() #显示图像
```

alpha0.1的新Ridge/转换器/标准化器及Pipeline对象引用均正确，仍degree12/include_biasFalse、只fit原训练组；train/val预测、reshape返回值保存、MAE配对和新网格预测均一致用_12_a0p1。散点、英语标题/单位/来源标识也正确。学生主动写到评分/绘图，不将扩展任务本身当错误；尚未提供运行输出，以上仅静态正确。

唯一明确问题在第二条plt.plot：纵坐标仍为y08_plot_pred_12_a1，label却是Degree12 Ridge alpha=0.1。label只改文字，不会自动切换曲线数据源；应核对刚生成的alpha0.1网格预测与这条曲线是否一致。第一条OLS基线y08_plot_pred_12是有意保留的未正则对照，正确且不需改。

当时仅提示绘图引用修正；最新存盘/输出见下，保留当时未收状态。

本次仅聊天源码静态审查，没有alpha0.1执行/报错/保存/预测数值或MAE证据，不登记新分数、图像或实际误绘，也不补跑教师参考模型。若旧_a1变量可用，这类图例身份错配可能不报错，但本次不推断实际内核状态。原alpha1与OLS成果保留、全部材料数据synthetic data；固定划分/特征流程/预定alpha范围不变，测试仍留最后一次评价。

## 2026-10-27：alpha0.1修正已保存且有实际运行输出

学生原答“已保存”。教师只读确认第二条plot已改为y08_plot_pred_12_a0p1，label对应alpha0.1，OLS基线仍y08_plot_pred_12。Notebook185387字节、mtime2026-10-27 21:16:34 +08:00、SHA256 14F5014153A8EA602E72C53E05991F66457BE3070003EE751513338FBB3D4200；代码计数13/14/15/16/31/49/null，Ridge格[49]保存了两个候选的顺序stdout与两幅PNG，无stderr/error。

修正后的绘图语句摘录：

```python
plt.plot(X08_plot["ZrB2_wt_pct"], y08_plot_pred_12_a0p1, label="Degree12 Ridge alpha=0.1")
```

原OLS基线保留，记提示后修正，不回写第13项首稿为正确。alpha1原stdout保持(30,1) (10,1)、训练MAE5.110107645592669、验证MAE4.781618618396107、网格(200,1)；alpha0.1新stdout为(30,1) (10,1)、训练MAE4.093727746977597、验证MAE4.146481680127238、网格(200,1)。数值为学生实际已保存输出，MPa由教师根据未缩放UTS补充，不是教师运行/预填。

新候选真实stdout：

```text
(30, 1) (10, 1)
4.093727746977597
4.146481680127238
(200, 1)
```

直接核对新内嵌PNG571×455/40357字节，训练/验证点、OLS12和Ridge alpha0.1英文图例、synthetic data标题及轴单位正确。alpha1旧图571×455/39905字节、原三模型图及旧OLS MAE仍保留。

| 十二次候选 | 训练MAE（MPa） | 验证MAE（MPa） |
| --- | ---: | ---: |
| 未正则化LinearRegression | 2.6663586323270865 | 4.190316693931817 |
| Ridge，alpha=1 | 5.110107645592669 | 4.781618618396107 |
| Ridge，alpha=0.1 | 4.093727746977597 | 4.146481680127238 |

两个Ridge候选同格各自新建工具，只fit固定训练组，再预测train/val与同一训练范围网格；网格不fit。无其他预定alpha、RMSE或测试预测评分，最终测试格仍未填。教师仅只读核验与原字节归档，没有执行/编辑模型或图像，不据计数推断重启/具体操作顺序或证明干净Run All。全部synthetic data；当前候选比较不证明全局最优、稳定优势或真实材料规律，不换seed/数据挑预期结果。

原Notebook字节直接归档，不教师拟合复算。当前只让学生比较同一验证列，题面见quizzes；没有新候选或测试任务。

## 函数求助、首次调用与静态检查（2026-10-28）

已试候选比较原答：“目前alpha为0.1时，表现效果最好”，按固定验证MAE判断正确，仅限已试候选。

教师布置管道工厂函数后，学生原答“不会”。以下是教师直接提供的完整小函数，属于代码帮助，不是学生独立实现或实际运行证据：

```python
def make_ridge08_pipeline(alpha):
    poly = PolynomialFeatures(degree=12, include_bias=False)
    scaler = StandardScaler()
    ridge = Ridge(alpha=alpha)

    pipe = Pipeline([
        ("poly", poly),
        ("scale", scaler),
        ("reg", ridge),
    ])
    return pipe
```

教师解释def、缩进、return及Ridge(alpha=alpha)的左右含义。学生对是否训练原答：“尚未训练的管道，这套流程返回值时一条新的pipe”。判断正确，教师补足.fit才执行训练的依据。

下一任务：调用函数创建alpha=0.01的pipe08_12_a0p01，并用固定X08_train/y08_train训练；只写两行先检查。学生首次原文（保留第二行4空格缩进及字母o）：

```python
pipe08_12_a0p01 = make_ridge08_pipeline(0.1)
    pipe08_12_aop1.fit(X08_train,y08_train)
```

函数调用/赋值和训练X/y选择方向正确；问题为传入0.1而非0.01、训练对象拼为pipe08_12_aop1而非创建对象、函数外第二行多余缩进。提示仅核对这三处并重发两行，未提供完整修正代码。

目前仅聊天文本静态审查，没有新截图/输出或存盘声明；不声称实际IndentationError、NameError、成功拟合或alpha0.01结果。未读取/运行/编辑或上传本地Notebook；既有alpha1/0.1真实结果和图像仍以上一核实版本为依据。测试继续保留，所有材料数据为synthetic data。

## 第14项提示后修正、alpha0.01实际预测保存（2026-10-28）

学生修正两行原文：

```python
pipe08_12_a0p01 = make_ridge08_pipeline(0.01)
pipe08_12_a0p01.fit(X08_train,y08_train)
```

传入强度、创建/fit对象名及顶层缩进均已修正；训练数据仍正确。记提示后修正，不覆盖首稿。教师检查通过后由学生运行，未代写整段候选比较。

新截图abde6a95-4044-42fc-bbd6-71dfecb65137.png中的预测源码：

```python
y08_train_pred_12_a0p01 = pipe08_12_a0p01.predict(X08_train)
y08_val_pred_12_a0p01 = pipe08_12_a0p01.predict(X08_val)
print(y08_train_pred_12_a0p01.shape)
print(y08_val_pred_12_a0p01.shape)
```

学生报告“输出为(30,)(10,)”。磁盘保存的真实stdout：

```text
(30,)
(10,)
```

预测首次提交的完整Pipeline、原始训练/验证输入及候选后缀均正确。原生1D是正常单目标输出，不要求重训或手动展开预测输入。为保持二维列约定，下一步仅学生自行保存reshape(-1,1)返回值并打印两shape；尚未提交，不预填完成。

教师只读磁盘：186509字节，mtime2026-10-28 11:07:09 +08:00，SHA256 D613D95966BC800A5BDDC797B96B64B2B0821137AB4686F2334A673D2AE03D13；代码计数2/3/4/5/6/9/8。Ridge格[9]同格保留alpha1和0.1的完整流程及原stdout/图，随后定义工厂、创建alpha0.01、只fit固定训练组，再完整管道预测train/val并打印原生shape。当前没有alpha0.01 reshape/MAE/网格图、RMSE或其他alpha结果；最后测试格[8]只有未开放测试的注释、outputs为空，不能由计数认定测试已评价。无保存error输出，不据计数推断重启/操作顺序或证明干净Run All。教师没有执行/编辑模型、Notebook或生成参考输出。

原五个已评分候选的MAE及原三模型图/两张Ridge图保留，无新指标，数据仍synthetic data。本次归档Notebook为学生已保存的原字节，不是教师重跑或改写；正式单元评估仍待。

## alpha0.01形状已保存、MAE已截图运行（2026-10-28）

截图2b535896-814f-4811-b07f-d2338e097802.png中两行reshape均正确保存回原变量，后续两print实际为(30,1)/(10,1)：

```python
y08_train_pred_12_a0p01 = y08_train_pred_12_a0p01.reshape(-1, 1)
y08_val_pred_12_a0p01 = y08_val_pred_12_a0p01.reshape(-1, 1)
print(y08_train_pred_12_a0p01.shape)
print(y08_val_pred_12_a0p01.shape)
```

记沿用已有方法的正确迁移，未补造原生1D错误或要求重新训练。

截图70da9833-e3e8-4b5a-a95d-33a99ed93bd1.png中首次MAE提交的有效语句如下（只摘录语句，截图中文注释不影响调用）：

```python
mae08_train_12_a0p01 = mean_absolute_error(y08_train,y08_train_pred_12_a0p01)
print(mae08_train_12_a0p01)
mae08_val_12_a0p01 = mean_absolute_error(y08_val,y08_val_pred_12_a0p01)
print(mae08_val_12_a0p01)
```

截图底部真实输出：

```text
3.7427907336170696
4.209376817121404
```

分别为alpha0.01的训练/验证MAE，单位MPa；真实值与预测值、后缀及打印对象首次均正确。学生原答“单位Mpa”，教师仅规范大小写为MPa。截图中更上方的两组MAE/图仍属alpha1/0.1，不混归给本候选。

本轮只读磁盘186688字节，mtime2026-10-28 12:26:39 +08:00，SHA256 56309200F2D53410D4E33C567EF587806F66B728D0325F3926A0E028E79E8283，代码计数2/3/4/5/6/10/8。Ridge格[10]已存正确reshape及实际(30,1)/(10,1)，末尾仅#计算MAE注释，尚无alpha0.01 MAE语句或数值。截图评分确有源码和输出，与磁盘保存阶段不同，不编造异常或归因；不称评分已存盘/已核对Notebook内数值。当前仅更新四档/progress，不上传这个尚无新MAE的Notebook作为评分完成件；等学生保存后再回读。

当前等待学生保存Notebook，并回答alpha0.1→0.01的训练/验证比较题；尚无新答复或保存声明。原固定训练/验证分组、仅训练fit的工厂与流程、五组旧MAE和三幅旧图均保留。最后[8]格仍仅封存测试的注释、outputs为空，无测试预测/评分；未见alpha0.01新图、alpha10/100/1000或RMSE。所有材料数据synthetic data；教师没有执行或编辑模型/Notebook，不据计数推断重启/操作顺序或干净Run All。

## MAE与打印循环保存闭环（2026-10-28）

先前alpha0.01评分仅截图的待办现已闭环：最新磁盘含正确两项MAE源码及实际3.7427907336170696/4.209376817121404，单位MPa。原来仅#计算MAE注释的存盘阶段保留为历史，不再让学生重算正确评分。

泛化比较学生原答：“不能，虽然训练MAE下降，但是验证MAE上升”，本次正确。

随后for任务学生答“不会”，教师直接提供如下完整小练习；这是代码帮助，不是学生独立实现：

```python
alpha_candidates08 = [0.01, 0.1, 1, 10, 100, 1000]

for alpha in alpha_candidates08:
    print(alpha)
```

教师解释列表、逐项取值、冒号和缩进。学生截图e86bc74a-2a15-4f57-9bdb-02aea77bd0d3.png显示代码与实际输出，最新存盘stdout也有：

```text
0.01
0.1
1
10
100
1000
```

归档初读12:48:39版本仅保存循环源码、尚无六值stdout；之后12:50:39版本已包含六值，现以新版本为准，不把初读差异当学生报错或未执行。

最新Notebook187400字节，mtime2026-10-28 12:50:39 +08:00，SHA256 D66F7883A4B57E0587332262765C37D6F16E0186AE1FCDA9EE0074B181C2A50D，代码计数2/3/4/5/6/12/8。Ridge格[12]保存正确工厂/alpha0.01仅训练组fit、完整trainval predict/reshape/MAE，最后stdout为(30,1)、(10,1)、3.7427907336170696、4.209376817121404，再顺序六值0.01/0.1/1/10/100/1000，无stderr/error。教师未执行或编辑Notebook/模型，不据计数推断重启/操作顺序或干净Run All。

当前for循环体仅print，没有新候选创建、fit或评分；10/100/1000尚无模型结果，六值打印不代表六种强度已评分。同格前面仍保留原三个Ridge候选训练/预测/旧图；原OLS、Ridge1/0.1与新0.01 MAE均保留。测试格[8]仍只有封存注释、outputs为空，没有测试评价；无RMSE或alpha0.01新图。全部synthetic data，正式评估及单元完成仍待。

下一小步仅请学生保留现有列表和循环内print，在循环内部用make_ridge08_pipeline(alpha)创建新的临时管道pipe08_candidate，再只fit固定X08_train/y08_train；先提交代码检查，暂不预测或评分。尚无学生提交，不预填实现或输出。

## 六候选循环完整源码与真实输出（2026-10-28）

学生先正确写出循环内创建/仅训练与train/val预测；再正确迁移reshape和MAE配对。训练指标起初名mae_train_candidate是合法且计算正确的变量，教师仅请求与mae08_val_candidate的命名约定一致，学生最终统一。没有因这一名称差异虚构报错或旧值混用。

最终完整循环原文：

```python
#用for循环减少重复代码
alpha_candidates08 = [0.01, 0.1, 1, 10, 100, 1000] #创建列表

for alpha in alpha_candidates08:
    pipe08_candidate = make_ridge08_pipeline(alpha) #创建新管道
    pipe08_candidate.fit(X08_train,y08_train) #训练这条管道
    y08_train_pred_candidate = pipe08_candidate.predict(X08_train)
    y08_val_pred_candidate = pipe08_candidate.predict(X08_val)
    y08_train_pred_candidate = y08_train_pred_candidate.reshape(-1, 1)
    y08_val_pred_candidate = y08_val_pred_candidate.reshape(-1, 1)
    mae08_train_candidate = mean_absolute_error(y08_train,y08_train_pred_candidate)#计算训练集MAE:结果变量 = 函数名称(真实值, 预测值)
    mae08_val_candidate = mean_absolute_error(y08_val,y08_val_pred_candidate)#计算验证集MAE
    print(alpha,mae08_train_candidate, mae08_val_candidate) #用for循环逐个打印这些值
```

教师静态检查通过后请学生运行并保存。学生回传且磁盘实际保存的六行输出（alpha、训练MAE、验证MAE；后两列MPa）：

```text
0.01 3.7427907336170696 4.209376817121404
0.1 4.093727746977597 4.146481680127238
1 5.110107645592669 4.781618618396107
10 6.706553867112996 6.599810058231322
100 8.17530755975614 7.289012030272067
1000 9.375614450668111 7.92572761478454
```

最新Notebook188383字节，mtime2026-10-28 13:22:13 +08:00，SHA256 0C9E29C365BEFF9A13EFFE36AEF87F284B02A7361FD292D7332E654C047EB11E，代码计数2/3/4/5/6/20/8。Ridge格[20]实际保存完整循环与以上六行stdout，和学生文字逐项一致，无stderr/error。每轮调用工厂新建degree12/include_bias=False、StandardScaler和相应Ridge，只fit固定训练组；完整Pipeline接收原始train/val输入，reshape返回值保存、两MAE与print均配对正确，预定分组和候选范围未变。

函数及for基础打印框架由教师在求助后提供；完整评分循环是分步任务下由学生补全、检查后执行，不算完全无帮助独立实现。六个强度真实评分现已完成，不再把10/100/1000写为未运行；尚无RMSE/独立测试。

当前问题：“六个候选中，应该优先保留哪个alpha？请依据当前固定验证集的MAE说明理由。”尚未作答，不把打印输出本身当作学生已经正确选模。

原三组OLS、手动Ridge alpha1/0.1/0.01的代码/真实结果及原三幅图保留；循环未新增绘图或RMSE。测试格[8]仍只有封存注释、outputs为空，没有测试预测/评分。教师只读核验、未代跑/改写Notebook或拟合参考模型；不据计数推断重启/具体操作顺序或证明干净Run All。全部材料数据synthetic data，不能据此认定真实最佳ZrB2含量或稳定最优强度，单元尚未完成。

## alpha0.1明确模型与预测首稿（2026-10-28）

选模真实原答“优先保存alpha = 0.1，当前的验证MAE最低”正确；循环末模型身份原答“1000，整个循环从0.01开始到1000结束”正确。属于当前对照题/状态检查，不升级正式评估完成或覆盖旧首错。

学生创建选中配置的两行原文正确：

```python
pipe08_selected = make_ridge08_pipeline(0.1)
pipe08_selected.fit(X08_train,y08_train)
```

教师检查通过后请其运行，并自行写两预测/二维整理/shape打印。学生聊天首次四行原文：

```python
y08_train_pred_selseted = pipe08_selected.predict(X08_train)
y08_val_pred_selected = pipe08_selected.predict(X08_val)
y08_train_pred_selected = y08_train_pred_selected.reshape(-1, 1)
y08_val_pred_selected = y08_val_pred_selected.reshape(-1, 1)
```

训练predict的保存名selseted与后续selected不同，无法把后续reshape连接到刚生成的这一预测；完整Pipeline和原始X输入本身正确。聊天验证两行配对正确。尚未给出两个print或实际shape。

磁盘另外一处差异，最后一行实际为：

```python
y08_val_pred_selected = y08_val_pred_candidate.reshape(-1, 1)
```

该右侧仍用循环临时候选预测，按当前循环最后对应alpha1000；不是刚生成的selected验证预测。只指出源码来源问题，不断言已经实际覆盖或生成错误分数。

当前磁盘224663字节，mtime2026-10-28 13:50:14 +08:00，SHA256 A7AAFE26E363D8AE8F74FE04CC6788200C77F5FEEAB6D75F179D6B802E36226C，计数2/3/4/5/6/22/8。存有selected正确创建/训练源码及Ridge(alpha0.1) Pipeline表示；也存有上述预测源码差异，但没有selected的shape/MAE/RMSE或相关错误输出，不能由同格execute_result认定后加预测/reshape已按当前源码成功执行，也不能编造NameError或推断内核旧变量是否存在。

最小提示：沿训练和验证两条数据链检查“predict保存名→reshape右侧→reshape保存名”是否逐字一致；统一selected拼写，磁盘验证reshape也接回本次selected预测，再补两项shape打印。先发修正段检查，暂不评分；尚未收到，不预填修正或实际输出。

原六强度循环及旧OLS/Ridge结果、图均保留。末测试格[8]仍仅注释、outputs为空，无测试预测评分。教师没有执行/改写模型或Notebook；本次只更新四档/progress，不把含待纠正源码的Notebook作为预测完成件上传。全部synthetic data，单元未完成。

## selected预测修正与真实shape保存（2026-10-28）

学生提示后修正训练拼写及本地验证来源，最新真实保存代码：

```python
y08_train_pred_selected = pipe08_selected.predict(X08_train)
y08_val_pred_selected = pipe08_selected.predict(X08_val)
y08_train_pred_selected = y08_train_pred_selected.reshape(-1, 1)
y08_val_pred_selected = y08_val_pred_selected.reshape(-1, 1)
print(y08_train_pred_selected.shape,y08_val_pred_selected.shape)
```

学生文字回报：

> (30,1),(10,1)

实际保存stdout末尾：

```text
(30, 1) (10, 1)
```

两预测都来自选中的alpha0.1完整管道，形状与对应目标一致，reshape保存返回值正确。一个print输出两个shape合法；第15项记提示后修正并实际保存，原selseted聊天稿及磁盘candidate差异仍保留。

最新Notebook188961字节，mtime2026-10-28 14:02:57 +08:00，SHA256 9F0C21A864934C40C4443B1C5321AA4D3566CEC706FC0DAC23D2D1E1149D94ED，代码计数2/3/4/5/6/23/8。Ridge格[23]末尾明确创建alpha0.1的pipe08_selected，只fit固定训练组，再由完整管道预测原始train/val，两reshape均读写对应selected变量；同一print输出两shape，实际最后stdout为(30, 1) (10, 1)。与学生文字报告一致，无stderr/error。

原六候选MAE、OLS/手动Ridge及三幅旧图均保留；selected尚无新MAE/RMSE或新图。测试格[8]仍仅封存注释、outputs为空，无测试预测评分。教师只读核验，未执行/编辑学生模型/Notebook或代拟合参考指标，不凭计数推断重启/操作顺序或干净Run All。全部synthetic data，单元未完成。

下一小步：教师只介绍RMSE的平方→平均→开方及本地可用的root_mean_squared_error导入/真实值在前的调用约定；学生自行用现有selected预测计算训练/验证RMSE，保存为rmse08_train_selected、rmse08_val_selected并打印，先发代码检查。没有学生提交或实际RMSE，不预填数值/通过，不使用测试集选模。

## selected RMSE首次代码正确、真实输出保存（2026-10-28）

教师仅提供新API导入和真实值在前的调用约定，学生首次有效代码（末尾“运行结果为……”按聊天叙述分开，不误判成Python源代码）：

```python
from sklearn.metrics import root_mean_squared_error
rmse08_train_selected = root_mean_squared_error(y08_train,y08_train_pred_selected)
rmse08_val_selected = root_mean_squared_error(y08_val,y08_val_pred_selected)
print(rmse08_train_selected,rmse08_val_selected)
```

学生真实报告、磁盘同样保存：

```text
5.07681785879906 4.870264282719038
```

分别为选中degree12/Ridge alpha0.1的训练/验证RMSE，单位MPa。没有调用循环末candidate，没有重拟合验证/测试或新增候选；此API实现首次配对正确，不升级先前工厂/循环框架为全部独立。

最新Notebook189333字节，mtime2026-10-28 14:52:35 +08:00，SHA256 D6C5FA365CEA729E542B6E58AEEFA85CD392FCD451553B74112E5FA930193A58，代码计数2/3/4/5/6/28/8。Ridge格[28]仍明确degree12、alpha0.1、仅原训练组fit，完整selected预测/reshape正确；末尾两项RMSE调用真实值与selected预测配对正确，保存stdout最后为(30, 1) (10, 1)，随后5.07681785879906 4.870264282719038。与学生回传完全一致，无stderr/error。

原OLS/手动Ridge/六强度MAE和三幅图均保留，无新增绘图；测试格[8]仍仅封存注释、outputs为空，无测试预测评分。教师只读核验，没有执行或编辑模型/Notebook及参考计算，不凭计数推断重启/操作顺序或干净Run All。全部synthetic data，当前候选选择只限既定训练/验证证据。

训练/验证实践阶段已完成；正式8题及独立最终测试仍待。正式单元末评估共8题：3概念、2数学、2Python、1材料迁移，逐题提出，保留首次原答/提示/修正，不能用此前教学检查替代。当前首题为：“当前选中模型的训练RMSE为5.0768 MPa，验证RMSE为4.8703 MPa。验证RMSE低于训练RMSE，能否据此排除过拟合？为什么？”尚无学生答复，不预填正确/错误/通过或总分。正式评估与代码检查结束、配置冻结后，再由学生在最后一次干净顺序运行中完成独立测试评价；现在不开放测试。

## 正式C1概念纠错阶段（2026-10-28）

本次没有新增Python任务或执行。学生C1原答与正式评价见quizzes及mistakes第16项：定义正确、绝对排除判断证据不足，已给最小覆盖提示但未收到修正；不能从已有RMSE输出自动登记C1通过。

正式已答1/8、通过0/8，当前仍C1，不进入C2或测试评价。本轮仅概念答复与教学记录，无新代码、运行或指标，没有执行/修改/上传Notebook或使用独立测试。此前RMSE等真实保存结果仍以b66086d8790e2e67c59ecd33650c5fe71478ed07的实践快照为依据，全部synthetic data；单元未完成。

## 正式C1纠正完成（2026-10-28）

本轮仅知识答复，无Python执行。C1覆盖范围提示后修正版已正确，全文存quizzes及mistakes第16项；教师仅精确“评价可靠”的措辞，记提示后完成，不新增程序/概念错误。正式已答1/8、通过1/8，C2首次待答。

本轮无新增代码、执行、指标或Notebook操作；训练/验证成果仍以上次已核实实践快照为依据，未使用独立测试。全部synthetic data，后续正式题、配置冻结后的最后一次独立测试与保存验收仍待，单元未完成。

## 正式C2概念检查通过（2026-10-28）

高偏差/高方差两情境的首次判断及原因正确，完整原答存quizzes。无新增Python执行，也不以本次口答替代P1/P2的真实代码和执行。当前正式已答2/8、通过2/8（首次正确1、提示后通过1），第3题C3待首次作答。

本轮只记录概念答复，无新代码、运行、指标或Notebook操作；此前实践快照和全部首错/帮助层级保留。题目为教学假设，材料数据synthetic data；独立测试与后续正式题仍待，单元未完成。

## 正式C3标准化流程首次正确（2026-10-28）

学生正确判断全部数据先fit不合适、训练组学习统计量、验证/测试transform，原答全文存quizzes。首次通过；本轮没有执行Python或改变Pipeline，不把口答代替后续P1/P2的真实代码/输出验收。

正式已答3/8、通过3/8（首次正确2、提示后通过1），M1数学题待首次作答。本轮无新代码、模型拟合、指标或Notebook操作。后续M1仅纸上理解，不把四次多项式加入既定候选搜索；原实践与全部首错/帮助层级保留，数据synthetic data，独立测试仍封存，单元未完成。

## 正式M1数学理解完成（2026-10-28）

首次1/4/5三计数正确；原题两项未答说明经原样提醒后，学生正确区分参数线性与输入曲线，并以参数增加解释复杂度。完整两段原答存quizzes，分段完成过程保留；没有新增模型运行。

三项数量首次均正确；对参数线性与输入曲线形状的核心区分正确，用参数增加解释复杂度的方向成立。仅重复原题尚未回答的两项要求，没有提供公式或原因等追加内容提示，记分段作答、无追加内容提示完成。‘线性图像更复杂’由教师规范为‘模型能表达的曲线更灵活’，不要求实际曲线一定更曲折，不另立错误。学生未逐字展开每个系数一次进入的数学式，不归档成其已经逐项说明该细节；本题基础通过，不等同长期数学表达熟练。

正式已答4/8、通过4/8（无内容提示完成3、提示后完成1），M2待首次作答。本轮仅纸上理解，没有新代码、运行、指标或Notebook操作，未将四次模型加入候选搜索。此前实践、M1早期数量首错、C1首答及提示层级均保留；材料数据synthetic data，后续正式题/最终测试仍待，单元未完成。

## 正式M2完成、P1独立代码待交（2026-10-28）

学生四个L1/L2惩罚及alpha强度/验证表现解释首次正确，全文存quizzes。正式已答5/8、通过5/8（无内容提示完成4、提示后完成1），开始第6题P1：

> 在最终测试格之前新增一个独立代码单元，不调用make_ridge08_pipeline。自行创建poly08_check（degree=12、include_bias=False）、scaler08_check（StandardScaler）、ridge08_check（alpha=0.1），组成新的pipe08_check。只用现有X08_train/y08_train拟合，对原始X08_val预测，保存为y08_val_pred_check并打印shape。保留原模型，先发完整代码检查，暂不运行或使用测试集。

尚未收到学生P1源码或执行结果，不将以前工厂/循环练习冒充本题独立实现。P1原生单目标1D或正确整理的2D预测均可，不强制reshape才算正确；所有工具须新建，不复用原模型的已拟合对象。新增代码单元须位于封存测试格之前，后续核验应按内容扫描单元，不固定旧的测试cell索引。

本轮仅纸上计算和概念答复，无新模型、代码执行、指标或Notebook操作。既有实践及所有首答/帮助层级保留；后续P1仍同配置/固定分组，不增加新调参候选，材料数据synthetic data，最终独立测试与单元完成仍待。

## 正式P1首次代码与已保存旧变量shape（2026-10-28）

截图68717965-3214-45ab-bb5d-21e0a1f06fff.png及磁盘同版首稿，完整原文：

```python
poly08_check = PolynomialFeatures(degree = 12, include_bias = False)#新的多项式转换器

scaler08_check = StandardScaler()#新的标准化器

ridge08_check = Ridge(alpha = 0.1)#新的回归器

pipe08_check = Pipeline([
    ("poly", poly08_check),
    ("scale", scaler08_check),
    ("reg", ridge08_check),
])#按“多项式展开 → 标准化 → 回归”组装
pipe08_check.fit(X08_train, y08_train)

#Ridge预测
y08_val_pred_check= pipe08_check.predict(X08_val)
#为保持本单元的二维列约定，下一步只整理已有预测的形状,.reshape(-1, 1),-1 表示自动确定行数，1 表示一列
y08_val_pred_12_a0p1 = y08_val_pred_12_a0p1.reshape(-1, 1)
print(y08_val_pred_12_a0p1.shape)
```

实际保存stdout：

```text
(10, 1)
```

新degree12/include_bias=False转换器、StandardScaler、Ridge(alpha0.1)及Pipeline均为新对象，步骤引用、仅fit原训练组、完整管道接收原始X08_val预测并保存y08_val_pred_check均正确。错误仅在最后reshape和print仍指向旧y08_val_pred_12_a0p1；输出(10,1)来自被打印的旧变量，不能作为新预测shape的证据，也不能反向声称新预测数值错误。

最新磁盘190605字节，mtime2026-10-28 17:35:45 +08:00，SHA256 F862B710DFA51586C9500B8B676F88EADEE99D00A218DB02ECF7ECDEC86BCCD1，代码计数40/41/42/43/44/45/47/46。P1为JSON索引14的独立代码格[47]，在末格之前，源码和截图68717965-3214-45ab-bb5d-21e0a1f06fff.png同版，实际保存stdout为(10, 1)，无error。不能因有输出或计数推断新变量已核验或干净Run All。

测试边界更新：末JSON索引16格[46]虽然仍以封存测试注释开头，其下现含原selected alpha0.1仅训练fit、train/val预测/shape/RMSE，实际RMSE仍5.07681785879906 / 4.870264282719038。全Notebook未见X08_test预测/评分，不能把位置重排当成测试泄漏，也不能继续写末格纯注释/输出为空。原OLS/六候选结果及旧图保留，数据均synthetic data。

最小提示：对照刚生成的y08_val_pred_check，让最后的形状整理及打印都检查同一新变量；若保留reshape须保存返回值，原生Ridge1D本身也合法，不强制reshape。先发修正末尾代码，不代写完整修正版；尚未收到，不记P1通过。

教师本轮只读核验，没有执行或编辑Notebook/模型，也不把待纠正P1作为完成件上传。本次只更新四档/progress；正式P1待修正/真实输出，P2/T1与最终独立测试仍待，单元未完成。

## 正式P1纠正后的真实shape与P2入口（2026-10-28）

修正版末尾已统一引用新预测：

```python
y08_val_pred_check= pipe08_check.predict(X08_val)
y08_val_pred_check = y08_val_pred_check.reshape(-1, 1)
print(y08_val_pred_check.shape)
```

实际stdout：

```text
(10, 1)
```

这次输出对应新y08_val_pred_check；首次旧变量输出和完整原稿仍存第17项。P1提示后完成，不误称从头全部错误或首次无帮助通过；新管道/训练边界/原始输入预测本来正确。

最新Notebook190599字节，mtime2026-10-28 17:43:45 +08:00，SHA256 49F7E42FE7027F7A2CEBDD0F04893121FBAC80E57A941840D4C1882411073816，代码计数40/41/42/43/44/45/49/46。P1独立格JSON index14[49]新对象/管道、仅训练fit、原始验证预测仍正确；reshape赋值与print现都引用y08_val_pred_check，实际stdout(10, 1)，与截图26e249c9-630f-497f-bafb-7b329c9dc571.png同版，无error。

第7/8题P2：

> 使用已经训练好的pipe08_check，为两条新的synthetic查询构造单列DataFrame X08_query_check，列名ZrB2_wt_pct、数值0.6和1.4（wt.%）。通过完整管道预测，保存为y08_query_pred_check；打印输入shape、预测数值和预测shape，不重新fit或手动重复转换。这两条查询没有真实UTS，能否计算MAE，为什么？在P1后、独立测试前新增代码单元，先发代码和判断检查，暂不运行。

尚无代码/运行结果。两条新输入只作synthetic查询，不新增真实材料实验，也不打开保留测试集。原生单目标1D或正确整理后的2D输出均有效。

原OLS/手动Ridge/六候选MAE和旧图均保留。末JSON index16[46]仍在封存注释下含selected训练/验证预测及RMSE准备，输出仍5.07681785879906/4.870264282719038，不能称仅注释；未见独立测试预测评分。教师只读核验，未执行/编辑Notebook或参考模型，不凭计数推断重启/运行顺序或干净Run All。全部synthetic data，单元未完成。

## 正式P2求助后的首次代码与真实输出（2026-10-29）

P2此前学生原答“不会”，属于求助，教师只提供输入表两行代码（构造X08_query_check与打印输入shape），没有提供完整预测/打印/MAE答案。输入构造不能计为学生完全独立实现；这次首稿是在该帮助之后提交。“不会”和编号/位置澄清不另记程序错误。

教师提供的输入构造窄范围代码为：

```python
# synthetic data：两条教学查询输入
X08_query_check = pd.DataFrame({"ZrB2_wt_pct": [0.6, 1.4]})
print(X08_query_check.shape)
```

学生首次完整P2代码（保留首稿）：

```python
X08_query_check = pd.DataFrame({"ZrB2_wt_pct": [0.6, 1.4]})
print(X08_query_check.shape)

poly08_query_check = PolynomialFeatures(degree = 12, include_bias = False)#新的多项式转换器

scaler08_query_check = StandardScaler()#新的标准化器

ridge08_query_check = Ridge(alpha = 0.1)#新的回归器

pipe08_query_check = Pipeline([
    ("poly", poly08_query_check),
    ("scale", scaler08_query_check),
    ("reg", ridge08_query_check),
])#按“多项式展开 → 标准化 → 回归”组装
pipe08_query_check.fit(X08_train,y08_train)

y08_query_pred_check= pipe08_query_check.predict(X08_query_check)
y08_query_pred_check = y08_query_pred_check.reshape(-1, 1)
print(X08_query_check.shape, y08_query_pred_check)
```

学生实际stdout（截图5754b227-d459-42c0-841e-2fc4b01ed342.png与保存版一致；来自新pipe08_query_check，不冒称来自P1管道）：

```text
(2, 1)
(2, 1) [[499.58619272]
 [498.37685233]]
```

查询DataFrame列名ZrB2_wt_pct、数值0.6/1.4与输入shape(2,1)正确。学生新建degree12/StandardScaler/Ridge(alpha0.1)及pipe08_query_check，并fit原X08_train/y08_train后预测查询；流程可运行，但偏离本题“沿用已训练pipe08_check、不重新fit”的要求。fit仍只用原训练数据，不是用查询点训练，不据此判查询泄漏或预测数值错误。reshape为(2,1)合法；末print打印的是X08_query_check.shape和预测数值，没有打印y08_query_pred_check.shape，两个(2,1)均来自输入shape，不能冒充预测shape验收。MAE判断尚未作答，不预填其答案或通过。

本次最小提示：保留正确输入表；省去本格新建工具、组装新管道和fit，预测时沿用P1已经训练好的管道；补查预测数组自己的shape。先请学生提交修正后的P2代码，再核验其真实运行保存。暂不代写完整修正版或继续T1/独立测试。

正式已作答7/8（P2仅代码部分已答）、通过6/8；已完成题无内容提示完成4、提示后完成2（C1/P1）。P2含教师输入表代码帮助，整题尚未完成；后续不得记为完全首次独立实现。材料数据与查询均为synthetic data，预测值不是实验UTS或最佳成分证据。

只读保存版核验：Notebook 191971字节，mtime 2026-10-29 09:48:15 +08:00，SHA256 8CF05DF4BB131F02F923EBE932D80BDEA37EFE2DEC3B4879050DD7A693FE5183；代码执行计数2/3/4/5/6/7/8/18/10。P1现为[8]，P2为新增独立格[18]，位于Markdown“第7步：锁定方案后的一次最终测试”之前。计数是执行编号，不是题号，不据此推断干净Run All。未见error/stderr或独立测试预测/评分。教师未执行或编辑Notebook/模型，本次不上传待修正Notebook；远程Notebook仍为P1闭环版。

## 正式P2修正后的真实代码、输出与理解判断（2026-10-29）

```python
#P2
X08_query_check = pd.DataFrame({"ZrB2_wt_pct": [0.6, 1.4]})
print(X08_query_check.shape)

#使用p1创建的管道
y08_query_pred_check = pipe08_check.predict(X08_query_check)
y08_query_pred_check = y08_query_pred_check.reshape(-1, 1)
print(y08_query_pred_check.shape, y08_query_pred_check)
```

实际stdout（修正截图572db99c-0f39-4b17-8b2b-218c664a973a.png[38]与存盘一致）：

```text
(2, 1)
(2, 1) [[499.58619272]
 [498.37685233]]
```

P2现无新fit，使用P1已训练管道；原始查询输入、预测对象与shape检查对应一致，实际两预测值的单位为MPa。这是沿用已学规则的新样本推断，不是实验UTS或独立测试评分。

学生MAE判断首次原答：

> 不能，MAE根据其计算公式，需要预测UTS和真实UTS

MAE理解小项首次回答正确，明确需要预测与真实UTS的配对差值。P2代码已提示后修正并核验真实执行保存；输入表此前有教师窄范围代码示范，首稿重新建模/fit与未打印预测shape均经提示纠正，因此整题记提示后完成。保留首稿/帮助层级，不因最后小项首答正确而改为全程独立完成。

第18项完整首稿与教师输入代码帮助均保留。正式已答7/8、通过7/8，无内容提示完成4、提示后完成3（C1/P1/P2）。当前第8/8题T1：

> 能否把本实验synthetic曲线的峰值作为真实7085Al/ZrB2复合材料的最佳颗粒含量，并据此确定疲劳设计方案？请从数据来源、模型覆盖范围及预测目标三个方面解释。

T1尚无首次答复，不预填答案/通过；独立测试仍等正式题与代码检查完成及方案冻结后由学生最后一次评价。全部材料数据与查询均为synthetic data，单元尚未完成。

最新只读Notebook 191456字节，mtime 2026-10-29 10:09:15 +08:00，SHA256 03BEB3206D686B6C24EAFBF07FF4EEB79F2289BD2070AD79CF755A182AC3C5B7，代码计数29/30/31/32/33/34/35/38/37。P1为JSON index14[35]，P2为index15[38]，紧接index16 Markdown“第7步”；末index17[37]仍只有selected训练/验证准备和既有RMSE 5.07681785879906 / 4.870264282719038 MPa，无独立测试预测评分。原OLS、六候选MAE与旧图保留，未见error/stderr。教师只读核验并上传学生Notebook原始字节，没有运行/编辑Notebook或模型，不据执行计数声称干净Run All。
