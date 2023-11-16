FROM python:3.10-bookworm
WORKDIR /code

COPY psst psst
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e ./psst

CMD ["python3"]