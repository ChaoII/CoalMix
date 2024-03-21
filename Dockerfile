FROM ubuntu:22.04 AS build
WORKDIR /app
COPY ./requirements.txt /app
RUN apt update -y && apt install python3 -y && apt install libopenblas-dev -y && apt install cmake -y && apt install python3-pip -y && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip install --no-cache-dir -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

FROM ubuntu:22.04
RUN apt update && apt-get install python3 -y && apt install libopenblas-dev -y
COPY --from=build /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
EXPOSE 5051
COPY . /app
COPY ./mosek/ /root/mosek
VOLUME ["/opt/app","/app"]
WORKDIR /app
ENV PYTHONPATH=/usr/local/lib/python3.10/dist-packages
CMD ["python3 -m gunicorn -c gunicorn_config.py main:app"]
