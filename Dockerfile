FROM psstbase
WORKDIR /code

COPY polysoleval polysoleval 
COPY models-ts models-ts 
COPY ranges ranges

CMD ["python3"]