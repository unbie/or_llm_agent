@echo off
REM 一键运行 C101/C105/C201/C205 强对比批量实验

setlocal
cd /d "%~dp0"

echo ==============================================
echo Running strong LLM-ALNS batch for 4 instances
echo ==============================================

python run_llm_optimize_strong_batch.py --rounds 10 --iters 1500 --final-iters 3000 --eval-runs 3

echo.
echo Done. Check: experiments_llm_optimize\strong_batch_report.json
pause
endlocal
