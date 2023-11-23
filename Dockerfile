FROM psstbase
WORKDIR /code

COPY polysoleval polysoleval
COPY models-ts models-ts
COPY ranges ranges
COPY main.py main.py

CMD ["python3"]