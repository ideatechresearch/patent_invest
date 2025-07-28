#!/bin/bash

set -e  # 脚本中遇到错误就退出
AIGC_DEBUG=${AIGC_DEBUG:-true}#如果没有设置，默认关闭 debug

echo "Running with AIGC_DEBUG=$AIGC_DEBUG"
# 拷贝 requirements.txt
cp -f requirements.txt docker/requirements.txt

# 构建镜像
docker build -t aigc/fastapi ./docker

# 删除已有容器（如果存在）
docker rm -f aigc 2>/dev/null || true

# 启动容器
docker run -d --name aigc --network qdrant_net -p 7000:7000 -v "$(pwd):/usr/src/app" -w /usr/src/app -e TZ=Asia/Shanghai -e AIGC_DEBUG="$AIGC_DEBUG" aigc/fastapi

# chmod +x run.sh
#./run.sh
