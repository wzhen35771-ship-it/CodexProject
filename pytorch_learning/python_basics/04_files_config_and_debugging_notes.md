# 第 4 课：路径、配置、命令行与排错

本课用 Python 标准库处理训练脚本的外围工作。完成后，你应能解释“文件在哪里”“当前运行哪个 Python”“参数从哪里来”“错误在哪一行”。示例不会安装依赖，所有读写都发生在自动清理的临时目录。

## 1. 当前工作目录和脚本目录

`Path.cwd()` 是启动程序时的工作目录；`Path(__file__).resolve().parent` 是脚本所在目录。两者可能不同。执行 `python pytorch_learning/python_basics/04_files_config_and_debugging_example.py` 时，如果终端位于项目根目录，工作目录就是项目根目录。

```python
from pathlib import Path

script_directory = Path(__file__).resolve().parent
config_path = script_directory / "config.json"
```

`/` 在这里拼接路径，不做除法。`resolve()` 形成绝对路径。`__file__` 在普通脚本中可用，在交互式解释器或某些 Notebook 中不一定存在。先判断代码运行环境，再选择路径起点。

实验不要覆盖项目资料。示例使用 `tempfile.TemporaryDirectory()`，离开 `with` 后删除临时目录；你只需观察输出。

## 2. 字典与 JSON

Python 字典适合程序内访问，JSON 适合存储配置。`json.dumps()` 把对象转为字符串，`json.loads()` 把字符串读回对象。

```python
import json

config = {"batch_size": 4, "epochs": 2}
text = json.dumps(config, ensure_ascii=False, indent=2)
loaded_config = json.loads(text)
print(loaded_config["batch_size"])
```

文件读写明确指定 `encoding="utf-8"`。JSON 不是任意 Python 对象的保存格式：不能直接保存模型类或 Tensor。JSON 中数组读取后是 `list`，不要把 Python 的 `tuple` 原样恢复视为默认行为。模型权重保存会在 PyTorch 课程介绍。

## 3. 用 argparse 接收参数

```python
parser.add_argument("--batch-size", type=int, default=4)
arguments = parser.parse_args()
batch_size = arguments.batch_size
```

命令行参数是 `--batch-size`，对象属性是 `batch_size`。`type=int` 把文字转换为整数，但正数条件仍需检查。示例自定义 `positive_integer` 做这一步。

在项目根目录的 PowerShell 运行：

```powershell
python pytorch_learning/python_basics/04_files_config_and_debugging_example.py
python pytorch_learning/python_basics/04_files_config_and_debugging_example.py --batch-size 3
python pytorch_learning/python_basics/04_files_config_and_debugging_example.py --show-traceback
python pytorch_learning/python_basics/04_files_config_and_debugging_example.py --help
```

如果 `python` 不存在，使用你实际选择的 Python 可执行文件；PowerShell 调用带空格的可执行路径时，用 `& '完整路径'`。不要照抄一个并不存在的路径。

## 4. 阅读异常和 traceback

先看异常最后一行的类型和原因，再沿堆栈找自己的文件和行号。示例主动解析一段错误 JSON，并用 `except json.JSONDecodeError` 捕获；这是演示错误处理，程序可以继续执行。

不要用 `except Exception: pass` 隐藏错误。文件不存在时检查路径；`ModuleNotFoundError` 时检查解释器和环境；`ValueError` 时检查输入。安装依赖之前，先确认错误属于哪一类。

## 5. 保持解释器与依赖一致

`sys.executable` 显示正在运行此脚本的 Python。VS Code 右下角选择的解释器应与终端实际运行的解释器一致。

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
```

`python -m pip` 通过当前 Python 运行 pip，能避免裸 `pip` 指向另一个环境。`venv` 是隔离环境；创建它不等于已安装 PyTorch。需要使用虚拟环境时，可以手动执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -m pip --version
```

以上是说明命令，本课脚本不会创建环境或自动安装包。VS Code 中选 `.venv/Scripts/python.exe` 后，新开终端再核对 `sys.executable`。PyTorch 是否可用，在后续课程用同一解释器检查。

## 练习与验收

1. 从项目根目录运行示例，再进入 `python_basics` 目录运行同一示例。解释两个路径输出的变化。
2. 用 `--batch-size 3`，指出参数如何进入 JSON 配置。用 `--batch-size 0`，观察 argparse 如何拒绝输入。
3. 开启 `--show-traceback`，找到错误发生的文件、行号、异常类型，解释程序为何仍能结束。
4. 记录 `sys.executable` 和 `python -m pip --version`，判断二者是否来自同一环境。

验收记录至少包含一组实际命令与输出、路径差别说明、一个错误定位结果。没有安装 PyTorch 时，记录环境问题即可，不要求本课运行 Tensor。

<details>
<summary>参考答案：完成观察后再展开</summary>

1. 工作目录随终端位置改变；脚本目录由文件位置决定，运行同一文件时保持不变。
2. `parse_args()` 得到 `arguments.batch_size`，程序把它放进 `config["batch_size"]`。0 不满足正整数条件，因此不会进入示例主体。
3. 错误类型是 `JSONDecodeError`，原因是键后缺少合法值。它被指定的 `except` 捕获；traceback 用于观察，不会再次抛出错误。
4. pip 输出包含安装路径。优先用选定解释器的完整路径加 `-m pip --version` 检查，不能仅凭终端名称判断。

</details>

继续：[第 5 课：阅读训练脚本](05_training_script_reading_notes.md)。
