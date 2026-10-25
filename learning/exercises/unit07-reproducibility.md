# 单元07：版本化成果与复现说明

归档日期：2026-10-25。此处保存学生实际完成的Notebook、对应输入快照及复现所需说明。**全部材料CSV为synthetic data（模拟教学数据），不是7085Al/ZrB₂实验结果。** 课程人口/利润数据也不是材料数据。

## 文件清单

- [学生Notebook](unit07-linear-regression.ipynb)：原样保留代码、输出和两幅图，不替学生重写或清空输出。
- [课程输入](data/unit07-ex1data1.txt)：97行、2列、无表头；Notebook从头运行首先需要它。
- [synthetic材料输入](data/unit07-materials-synthetic.csv)：100行、3列、有表头；输入是ZrB₂含量、时效温度，目标是人为生成UTS。
- [课程数据来源](data/unit07-data-source.md)及[synthetic生成协议](data/unit07-materials-synthetic-data-source.md)。
- [依赖版本](requirements-unit07.txt)：从原运行环境读取的10个核心/运行工具版本；不是整个传递依赖树的哈希锁文件。
- 本目录的`.gitattributes`：让上述新增文本资产以LF换行检出，减少Windows换行转换带来的文件哈希差异。

不纳入版本管理的资产：`.venv`、本机内核注册文件、Jupyter运行状态、缓存、临时截图；不改原课程目录。本次新说明与依赖清单直接写入GitHub，不为归档另建本地副本。

## 运行环境与证据等级

原Notebook的Python版本为3.10.11，核心包是NumPy2.2.6、pandas2.3.3、matplotlib3.10.7、scikit-learn1.7.2；完整的本次核心版本见requirements。

学生已经按“重启内核→运行全部单元”操作并报告无报错；教师只读核实保存的6个非空代码格计数连续1—6、无error输出，结果与此前一致。教师未执行或修改这个Notebook。

原本地`.venv`使用了system-site-packages，不能把它当可移植环境。下面采用新的独立venv，不复用该设置。**本次没有在新建环境中重新执行Notebook**；因此新环境流程是复现步骤，不冒称已经完成另一轮干净环境验证。依赖清单未锁定每个传递依赖、系统库或构建哈希；跨系统浮点尾数/图像渲染也不保证逐位相同。

## Windows PowerShell：从干净检出启动

先安装Python3.10（原运行补丁版本3.10.11）。在仓库根目录执行以下命令；不要把已有代码或数据覆盖成另一份答案。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r learning/exercises/requirements-unit07.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name unit07 --display-name "Unit 07 (.venv)"
cd learning/exercises
..\..\.venv\Scripts\python.exe -m jupyterlab unit07-linear-regression.ipynb
```

在Notebook中选择新环境的`Unit 07 (.venv)`内核，再重启并从上到下运行全部单元。不要重新生成CSV或挑新的随机种子替代归档输入。

## macOS / Linux：同样的目录布局

已安装Python3.10后，从仓库根目录：

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r learning/exercises/requirements-unit07.txt
.venv/bin/python -m ipykernel install --prefix .venv --name unit07 --display-name "Unit 07 (.venv)"
cd learning/exercises
../../.venv/bin/python -m jupyterlab unit07-linear-regression.ipynb
```

## 工作目录与常见问题

Notebook实际读取的是`data/unit07-ex1data1.txt`和`data/unit07-materials-synthetic.csv`。这些路径相对于内核当前工作目录，不是任意终端所在目录。因此按上述命令在`learning/exercises/`启动、在相应Notebook目录运行，不要移动Notebook却不保留相邻的`data/`。

metadata里的内核名是`unit07`；新电脑需要按上文重新注册，不能把旧电脑的内核配置文件提交后当成新环境。Notebook无需为新电脑修改绝对路径。

## 输入快照与校验

下表是仓库文本内容原始字节的SHA-256；如果人为改写数据、重保存Notebook或改变换行，哈希会变化。Notebook重跑后可能更新执行元数据/图像，哈希不再等于归档快照并不自动表示预测数值错误。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| unit07-linear-regression.ipynb | 77369 | B466122C36DBBFFA3C1B5BEFBD0A71BAEB3BE1D0D860063D7C2204DFA365F2DF |
| data/unit07-materials-synthetic.csv | 1840 | C13BA89B22067720CA8B5C9EBB760A69FCEC96B49BF9E58C453D62843D6A22E7 |
| data/unit07-ex1data1.txt | 1359 | E69559FAC1A2F68F704222629E603299BFA8AABFD88E81225E52E125484764A0 |

在仓库根目录可用PowerShell的`Get-FileHash learning/exercises/data/unit07-materials-synthetic.csv -Algorithm SHA256`检查CSV。归档数据本身才是本次运行的输入依据；生成协议解释出处，不要求每次训练重新抽取。

## 对照保存的输出

允许不同系统存在小的浮点尾数差异，比较数值时可使用约1e-6的绝对容差作为这份学习快照的检查尺度；这不是材料实验误差标准，也不承诺所有平台一定落在该容差内。

| 同一次划分的测试结果 | MAE | RMSE | R² |
|---|---:|---:|---:|
| 课程数据，seed42 | 2.495643609921872 | 3.8611253088773583 | 0.6555893767196204 |
| synthetic材料，seed42 | 5.927584783266172 MPa | 7.214085276099755 MPa | 0.7264239400097976 |

另一划分seed7的synthetic材料测试MAE为5.358826057468858 MPa；它不是seed42的结果，不能与上表另两项拼成同一组评估。

新输入`[0.50,140.0]`和`[1.50,170.0]`沿用seed7已训练的标准化器和回归模型，保存预测约493.83202391、503.35781219 MPa，shape为`(2,1)`。没有新输入的真实目标，未计算它们的预测误差。

这些是学生已有运行的对照值，不是本次教师新训练的参考模型。全部synthetic结果仅说明对人为教学生成关系的拟合，不验证真实材料规律或工艺可靠性。
