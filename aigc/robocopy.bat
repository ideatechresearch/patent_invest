@echo off
setlocal enabledelayedexpansion
REM ============================================
REM 同步当前目录文件到 Z:\Document\aigc
REM 仅覆盖目标中已有的文件，且必须比目标文件更新
REM ============================================

:: 源目录和目标目录
set "SRC=%CD%"
set "DST=Z:\Document\aigc"

echo 正在同步: %SRC% - %DST%
echo.

:: 遍历子目录，排除不需要的，并且只在目标中存在才复制
for /d %%D in (*) do (
    set "dirname=%%D"
    if /I not "!dirname!"=="docker" ^
    if /I not "!dirname!"=="docker_build" ^
    if /I not "!dirname!"==".venv" ^
    if /I not "!dirname!"==".idea" ^
    if /I not "!dirname!"==".git" ^
    if /I not "!dirname:~0,1!"=="." (
        if exist "%DST%\%%D\" (
            echo 更新目录: %%D
            xcopy "%%D" "%DST%\%%D\" /E /D /Y /I >nul
        ) else (
            echo 跳过目录: %%D （目标中不存在）
        )
    )
)

:: 复制当前目录下普通文件（非隐藏文件），前提是目标文件已存在
for %%F in (*.*) do (
    set "filename=%%F"
    if /I not "!filename:~0,1!"=="." (
        if exist "%DST%\%%F" (
            echo 更新文件: %%F
            xcopy "%%F" "%DST%\%%F" /D /Y >nul
        ) else (
            echo 跳过文件: %%F （目标中不存在）
        )
    )
)
REM robocopy "%SRC%" "%DST%" *.* /E /XO /COPY:DAT /IS /XC /XN /XX /R:0 /W:0 /LOG+:sync.log

echo.
echo 同步完成。
pause
