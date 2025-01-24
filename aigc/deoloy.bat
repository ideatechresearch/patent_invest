@echo off

copy /Y requirements.txt docker\requirements.txt

docker build -t aigc/fastapi ./docker

docker rm -f aigc 2>nul

docker run -d --name aigc --network qdrant_net -p 7000:7000 -v "%cd%:/usr/src/app" -w /usr/src/app -e TZ=Asia/Shanghai aigc/fastapi