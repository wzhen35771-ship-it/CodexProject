# 本机运行环境

核对日期：2026-10-16。目录整理未安装或升级依赖。

## Python 与 PyTorch

Python 先修示例只使用标准库。在项目根目录运行：

```powershell
python -X utf8 learning/python_basics/01_data_structures_example.py
python -X utf8 learning/python_basics/05_training_script_reading_exercise.py --check
```

本机已验证可用的 PyTorch 解释器是：

```powershell
$studyPython = 'C:/Users/admin/Documents/Codex/2026-10-12/ban-2/outputs/programming-starter/.venv/Scripts/python.exe'
& $studyPython -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
& $studyPython -X utf8 learning/pytorch_basics/07_training_example.py
```

该环境为 PyTorch 2.6.0+cpu，CUDA 不可用；不表示机器一定没有 GPU。路径在项目外，本次未迁移或删除。课程依赖记录为 [requirements.txt](../learning/requirements.txt)，复制项目到其他电脑后需重新选择解释器。

## PDF 与 Zotero

[zotero-env/.venv/](../zotero-env/README.md) 是已有 PDF 提取环境；虚拟环境中包含原路径，因此保留 zotero-env 目录。PDF 生成脚本使用 reportlab 和 Windows 微软雅黑字体，详见 [脚本说明](../scripts/README.md)。

Zotero 本地 API 为 http://127.0.0.1:23119；本地 API 配置开启后仍需 Zotero 桌面程序运行。无需为整理项目重新安装 Zotero。

## Codex 代理启动入口

入口已移动到 [start_codex_proxy.cmd](../scripts/start_codex_proxy.cmd)。代理默认是 http://127.0.0.1:7892，可通过 PowerShell 脚本的 -Port 参数修改。启动脚本不修改 Windows 全局代理或 Codex 用户配置。

运行脚本只会为新启动进程设置代理；已有 Codex 进程可能复用旧环境，需要退出后再打开。目录整理期间只验证脚本语法，不主动重启当前应用。
