@echo off
REM ============================================================
REM 学术论文Markdown转Word批处理脚本
REM 功能: 将ALNS_VRP_项目报告.md转换为符合学术规范的Word文档
REM ============================================================

echo.
echo ========================================
echo 学术论文文档格式化工具
echo ========================================
echo.

REM 检查Pandoc是否已安装
where pandoc >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未检测到Pandoc工具！
    echo.
    echo 请访问以下网址下载安装Pandoc:
    echo https://pandoc.org/installing.html
    echo.
    echo 安装完成后重新运行此脚本。
    pause
    exit /b 1
)

echo [1/4] 检测到Pandoc工具...
pandoc --version | findstr "pandoc"
echo.

echo [2/4] 开始转换Markdown文档...
echo 源文件: ALNS_VRP_项目报告.md
echo 目标文件: ALNS_VRP_项目报告_学术版.docx
echo.

REM 执行转换（使用学术模板参数）
pandoc "ALNS_VRP_项目报告.md" ^
    -o "ALNS_VRP_项目报告_学术版.docx" ^
    --from=markdown ^
    --to=docx ^
    --toc ^
    --toc-depth=3 ^
    --number-sections ^
    --highlight-style=tango ^
    --reference-doc=academic_template.docx ^
    --metadata title="基于大语言模型自动生成启发式算子的ALNS算法求解生鲜物流VRP问题" ^
    2>conversion_error.log

if %errorlevel% neq 0 (
    echo.
    echo [错误] 转换失败！错误信息:
    type conversion_error.log
    pause
    exit /b 1
)

echo [3/4] 转换成功！
echo.

echo [4/4] 生成的文档:
dir /b "ALNS_VRP_项目报告_学术版.docx"
echo.

echo ========================================
echo 转换完成！
echo ========================================
echo.
echo 生成的Word文档包含:
echo   - 自动生成的目录（3级深度）
echo   - 章节自动编号
echo   - 代码高亮显示
echo   - 表格和公式保留
echo.
echo 请打开Word文档进行最后的格式微调。
echo.
pause
