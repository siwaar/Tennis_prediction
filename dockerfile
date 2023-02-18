FROM python

WORKDIR /root/
COPY . /root/

RUN pip install -r requirements.txt
RUN pip install pre-commit &&  pre-commit install

RUN python3 code_inference.py
