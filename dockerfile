FROM python

ADD requirements.txt /

RUN pip install -r requirements.txt
#RUN pip install pre-commit &&  pre-commit install

ADD ./app
WORKDIR /app

EXPOSE 5000
CMD ["python3", "app.py"]
