@echo off
setlocal

:: 源目录和目标目录
set "SRC=%CD%"
set "DST=E:\Downloads\aigc"

echo 正在同步: %SRC% -> %DST%
echo.

:: 同步文件，排除当前脚本自身
robocopy "%SRC%" "%DST%" /E /XO /IS /XX /R:0 /W:0 /XF "%~nx0" /LOG+:sync.log

echo.
echo 同步完成。
pause

