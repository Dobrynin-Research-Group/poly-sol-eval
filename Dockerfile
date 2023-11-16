FROM psstbase
WORKDIR /code

COPY polysoleval polysoleval
COPY models_test models_test

CMD ["python3"]