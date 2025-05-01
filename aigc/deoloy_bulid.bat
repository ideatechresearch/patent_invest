@echo off
setlocal enabledelayedexpansion

REM 清空旧构建目录
rd /s /q docker_build
mkdir docker_build

REM 复制 Dockerfile 和 requirements.txt
copy /Y docker\Dockerfile docker_build\Dockerfile
copy /Y requirements.txt docker_build\requirements.txt

REM 复制项目代码，排除隐藏目录和指定文件夹
for /d %%D in (*) do (
    set "dirname=%%D"
    if /I not "!dirname!"=="docker" if /I not "!dirname!"=="docker_build" if /I not "!dirname:~0,1!"=="." if /I not "!dirname!"==".venv" if /I not "!dirname!"==".idea" if /I not "!dirname!"==".git" (
        xcopy "%%D" "docker_build\%%D\" /E /Y /I >nul
    )
)

REM 复制普通文件（非隐藏文件）
for %%F in (*.*) do (
    set "filename=%%F"
    if /I not "!filename:~0,1!"=="." (
        copy /Y "%%F" "docker_build\%%F" >nul
    )
)

REM 构建包含代码的镜像
docker build -t aigc/fastapi docker_build > build.log 2>&1
REM notepad build.log

REM 移除旧容器（忽略错误）
docker rm -f aigc 2>nul

REM 运行容器，继续挂载当前项目代码（方便开发热更新）
docker run -d --name aigc --network qdrant_net -p 7000:7000 -v "%cd%:/usr/src/app" -w /usr/src/app -e TZ=Asia/Shanghai aigc/fastapi