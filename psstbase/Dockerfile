FROM python:3.10-bookworm
WORKDIR /code

COPY PSST PSST
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e ./PSST

CMD ["python3"]