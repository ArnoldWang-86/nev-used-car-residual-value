# -*- coding: utf-8 -*-
r"""make_bat.py —— 生成可在中文 Windows 下正常双击运行的 .bat

关键点：cmd.exe 按系统 ANSI 代码页（中文 Windows = GBK/cp936）解析 .bat 文件，
      而不是按 chcp 设置。因此 .bat 必须用 GBK 编码写入；
      用 UTF-8 写会导致中文被拆成乱码命令，报"不是内部或外部命令"。
      chcp 65001 只影响后续输出，救不了解析阶段。
"""
import os

# --- 中文 Windows 控制台默认 GBK，强制 UTF-8 输出，避免打印特殊字符时崩溃 ---
try:
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BATS = {
    "查看采集进度.bat": r"""@echo off
setlocal
title 蔚来二手车项目 - 采集进度看板
cd /d "%~dp0"

rem GBK 写文件 + chcp 65001 显示：前者保证 cmd 能正确解析本文件的中文，
rem 后者保证 python 的 UTF-8 输出在控制台不乱码。两者缺一不可。
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ============================================================
echo   新能源二手车残值预测 - 采集进度看板
echo ============================================================
echo.
echo   每 30 秒自动刷新，直接关闭本窗口即可退出
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo   [错误] 找不到 python 命令，请确认 Python 已安装并加入 PATH。
    echo.
    pause
    exit /b 1
)

python src\progress.py --watch --interval 30

echo.
echo   看板已退出。
pause
""",
    "采集车源详情.bat": r"""@echo off
setlocal
title 蔚来二手车项目 - 抓取车辆详情(Phase B)
cd /d "%~dp0"

chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ============================================================
echo   抓取车辆详情（Phase B）
echo ============================================================
echo.
echo   限速 60 秒/请求，可随时关闭窗口中断，支持断点续传。
echo   想边抓边看进度，请另开"查看采集进度.bat"。
echo.
pause

python src\collect.py --phase b

echo.
echo   本次抓取已结束。
pause
""",
    "探测封禁状态.bat": r"""@echo off
setlocal
title 蔚来二手车项目 - 探测封禁状态
cd /d "%~dp0"

chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ============================================================
echo   探测采集封禁状态（只发 1 个请求）
echo ============================================================
echo.
echo   被封禁时请勿反复运行本脚本 —— 连续试探会延长封禁时间。
echo.

python src\collect.py --probe
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
    echo   结论：IP 状态正常，可以继续采集。
) else (
    echo   结论：仍被拦截，请等待数小时后再试。
)
echo.
pause
""",
}

for name, content in BATS.items():
    path = os.path.join(ROOT, name)
    # 换行用 CRLF，编码用 GBK —— 两者都是 cmd.exe 的解析要求
    data = content.replace("\n", "\r\n").encode("gbk")
    with open(path, "wb") as f:
        f.write(data)
    print(f"已生成 {name}  ({len(data)} 字节, GBK/CRLF)")
