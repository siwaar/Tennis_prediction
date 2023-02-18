FROM ubuntu:20.04

RUN apt update
RUN apt install python3-pip -y

WORKDIR /root/
COPY . /root/

RUN python -m venv tennis_env
RUN source tennis_env/bin/activate
RUN pip install -r requirements.txt
RUN pip install pre-commit &&  pre-commit install

RUN python code_inference.py
