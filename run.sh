export modelpath=models_test
export tmppath=tmp
export rangepath=psst/ranges
uvicorn polysoleval.main:app --port 8000 --reload