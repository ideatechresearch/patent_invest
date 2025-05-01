FROM python:3
#ENV OPENBLAS_NUM_THREADS=1
#ENV OMP_NUM_THREADS=1
#ENV MKL_NUM_THREADS=1

WORKDIR /usr/src/app
#COPY . .
COPY requirements.txt ./
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt

EXPOSE 7000
# CMD [ "python", "./main.py" ]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--workers", "4"]