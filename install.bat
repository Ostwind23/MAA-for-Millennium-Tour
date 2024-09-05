@echo off
chcp 65001 >nul
call conda activate MAA
python ./install.py
echo 自动打包命令执行完成，请按任意键结束
pause >nul
