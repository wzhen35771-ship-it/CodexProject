# 工具脚本

## Codex 代理启动

双击 [start_codex_proxy.cmd](start_codex_proxy.cmd)，或执行：

```powershell
powershell.exe -NoProfile -File scripts/start_codex_proxy.ps1 -Port 7892
```

脚本检查本机端口、定位已安装的 Codex，然后设置该进程的代理。默认端口统一为 7892；错误提示不绑定特定代理软件。本次未实际重启应用，不能据此声称重连问题已修复。

## 生成综述阅读笔记

[build_sgg_notes.py](build_sgg_notes.py) 是现有六页笔记的源脚本，输出为 [sgg_survey_notes.pdf](../output/pdf/sgg_survey_notes.pdf)。脚本使用项目相对位置定位输出，Windows 字体路径仍为系统字体目录。需要包含 reportlab 的 Python，例如此前使用的 bundled runtime：

```powershell
$pdfPython = 'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $pdfPython scripts/build_sgg_notes.py
```

运行会覆盖笔记 PDF；修改版面后应重新渲染检查。此次仅移动源脚本并修复输出路径，已验证的 PDF 原始内容保持不变。历史周报 PDF 是单独的快照，可编辑正文在 [周报](../docs/reports/weekly_report_2026_10_13.md)。
