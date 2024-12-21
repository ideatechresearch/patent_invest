FROM python:3

WORKDIR /usr/src/app


COPY requirements.txt ./
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt

EXPOSE 3300
CMD [ "python", "./flask_app.py" ]