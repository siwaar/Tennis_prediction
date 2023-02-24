FROM python

ADD requirements.txt /

RUN pip install -r requirements.txt

ADD . /app
WORKDIR /app

EXPOSE 5000
CMD ["python3", "app.py"]
