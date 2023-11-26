FROM jacobs27/psst-deps
WORKDIR /code

COPY polysoleval polysoleval
COPY ranges ranges
COPY models-ts models-ts
COPY main.py ./
