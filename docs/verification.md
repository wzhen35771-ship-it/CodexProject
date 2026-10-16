# 项目整理验证记录

日期：2026-10-16。使用现有 PyTorch 2.6.0+cpu 环境，未安装或升级依赖。命令为所选解释器加 `-X utf8 文件路径`；Python 综合练习另外加 `--check`。

## 实际运行

全部 14 个示例与两份已有答案的练习运行成功，共 16 项。其他练习包含空白或未完成内容，未据此宣称通过。运行成功不替代学习者的独立验收。

| 文件 | 结果 |
|---|---|
| `learning/python_basics/01_data_structures_example.py` | 通过 |
| `learning/python_basics/02_control_flow_and_functions_example.py` | 通过 |
| `learning/python_basics/03_classes_and_modules_example.py` | 通过 |
| `learning/python_basics/04_files_config_and_debugging_example.py` | 通过 |
| `learning/python_basics/05_training_script_reading_example.py` | 通过 |
| `learning/pytorch_basics/01_tensors_example.py` | 通过 |
| `learning/pytorch_basics/02_operations_example.py` | 通过 |
| `learning/pytorch_basics/03_shapes_example.py` | 通过 |
| `learning/pytorch_basics/04_autograd_example.py` | 通过 |
| `learning/pytorch_basics/05_models_example.py` | 通过 |
| `learning/pytorch_basics/05_parameter_update_example.py` | 通过 |
| `learning/pytorch_basics/06_data_example.py` | 通过 |
| `learning/pytorch_basics/07_training_example.py` | 通过 |
| `learning/pytorch_basics/08_evaluation_example.py` | 通过 |
| `learning/python_basics/05_training_script_reading_exercise.py` | 通过 |
| `learning/pytorch_basics/01_tensors_exercise.py` | 通过 |

第 7 课实际完成 200 轮、600 次更新，并通过原有损失/准确率断言；第 8 课通过保存加载前后 logits 一致性检查，输出位置随课程目录迁移到 learning/pytorch_basics/outputs/。这些仅验证人工教学数据流程，不证明真实 SGG 效果。

## 完整性与静态检查

- 课程及 PDF 生成脚本共 24 个 Python 文件通过语法解析。
- 代理 PowerShell 脚本通过语法解析；未启动或重启 Codex。
- 9 份个人练习和 12 份 PDF 共 21 个文件逐字节 SHA-256 一致。
- Markdown 本地链接 139 处核对通过，无断链。
- PDF 内容未重新生成，六页综述笔记继续使用此前已渲染检查的版本。

原始命令输出与哈希记录保存在本机忽略目录 .maintenance/；回退文本备份也在那里。
