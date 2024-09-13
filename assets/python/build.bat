@echo off
chcp 65001 >nul
call conda activate MAA
python ./build.py
echo 自动构建命令执行完成，请按任意键结束
pause >nul
